from __future__ import annotations

import math

from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import CancelAll, PlaceOrder, Side, StepState


def _std_normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _std_normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _inv_normal_cdf(p: float) -> float:
    if p <= 1e-10:
        return -6.0
    if p >= 1.0 - 1e-10:
        return 6.0
    if p < 0.5:
        return -_inv_normal_cdf(1.0 - p)
    t = math.sqrt(-2.0 * math.log(1.0 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)


def _sigma_ticks(p_frac: float, steps_remaining: int, sigma: float = 0.02) -> float:
    if steps_remaining <= 0 or p_frac <= 0.01 or p_frac >= 0.99:
        return 0.3
    variance = steps_remaining * sigma * sigma
    if variance < 1e-15:
        return 0.3
    sd = math.sqrt(variance)
    z = _inv_normal_cdf(p_frac)
    return max(0.2, 100.0 * sigma * _std_normal_pdf(z) / sd)


class Strategy(BaseStrategy):

    def __init__(self):
        self.fair_value: float | None = None
        self.prev_comp_bid: int | None = None
        self.prev_comp_ask: int | None = None

    def _update_fair_value(self, comp_bid, comp_ask, state: StepState):
        if comp_bid is not None and comp_ask is not None:
            raw_mid = (comp_bid + comp_ask) / 2.0
            if self.fair_value is None:
                self.fair_value = raw_mid
            else:
                if self.prev_comp_ask is not None and comp_ask > self.prev_comp_ask:
                    signal = (self.prev_comp_ask + comp_ask) / 2.0
                    self.fair_value = 0.7 * signal + 0.3 * self.fair_value
                elif self.prev_comp_bid is not None and comp_bid < self.prev_comp_bid:
                    signal = (self.prev_comp_bid + comp_bid) / 2.0
                    self.fair_value = 0.7 * signal + 0.3 * self.fair_value
                else:
                    self.fair_value = 0.6 * raw_mid + 0.4 * self.fair_value
        elif comp_bid is not None:
            self.fair_value = comp_bid + 1.0
        elif comp_ask is not None:
            self.fair_value = comp_ask - 1.0

        if self.fair_value is None:
            self.fair_value = 50.0

        if state.buy_filled_quantity > 0 and state.sell_filled_quantity == 0:
            self.fair_value -= 0.5
        elif state.sell_filled_quantity > 0 and state.buy_filled_quantity == 0:
            self.fair_value += 0.5

        self.fair_value = max(3.0, min(97.0, self.fair_value))
        self.prev_comp_bid = comp_bid
        self.prev_comp_ask = comp_ask

    def on_step(self, state: StepState):
        actions = [CancelAll()]

        comp_bid = state.competitor_best_bid_ticks
        comp_ask = state.competitor_best_ask_ticks

        self._update_fair_value(comp_bid, comp_ask, state)

        if state.steps_remaining < 20 or comp_bid is None or comp_ask is None:
            return actions

        obs_spread = comp_ask - comp_bid
        if obs_spread < 4:
            return actions

        fv = self.fair_value
        net_inv = state.yes_inventory - state.no_inventory
        skew = -net_inv * 0.015
        skew = max(-2.5, min(2.5, skew))
        fv_skewed = fv + skew
        fv_skewed = max(2.0, min(98.0, fv_skewed))

        p_frac = max(0.02, min(0.98, fv / 100.0))
        sig = _sigma_ticks(p_frac, state.steps_remaining)

        size_scale = max(1.0, 2.5 / max(0.3, sig))

        max_inv = 150.0
        skip_buy = net_inv > max_inv
        skip_sell = net_inv < -max_inv

        available_cash = state.cash

        late_scale = 1.0
        if state.steps_remaining < 100:
            late_scale = 0.5
        if state.steps_remaining < 40:
            late_scale = 0.25

        base_cushion = max(0.4, 0.9 * sig)

        if obs_spread >= 8:
            levels = [
                (1, base_cushion * 0.5, 18.0 * size_scale),
                (2, base_cushion * 0.5 + 1.0, 5.0 * size_scale),
                (3, base_cushion * 0.5 + 2.0, 3.0),
            ]
        elif obs_spread >= 6:
            levels = [
                (1, base_cushion * 0.7, 12.0 * size_scale),
                (2, base_cushion * 0.7 + 1.0, 3.0 * size_scale),
            ]
        elif obs_spread >= 4:
            levels = [
                (1, base_cushion, 8.0 * size_scale),
            ]
        else:
            levels = [
                (1, base_cushion * 1.5, 2.0),
            ]

        for offset, min_cushion, base_qty in levels:
            bid_tick = comp_bid + offset
            ask_tick = comp_ask - offset

            if bid_tick >= ask_tick:
                continue

            qty = max(0.01, base_qty * late_scale)
            qty = int(qty * 100) / 100.0

            if not skip_buy and 1 <= bid_tick <= 99:
                if fv_skewed > bid_tick + min_cushion:
                    bq = qty
                    cost = (bid_tick / 100.0) * bq
                    if cost > available_cash * 0.14:
                        bq = (available_cash * 0.14) / (bid_tick / 100.0)
                        bq = int(bq * 100) / 100.0
                    if bq >= 0.01:
                        actions.append(PlaceOrder(side=Side.BUY, price_ticks=bid_tick, quantity=bq))
                        available_cash -= (bid_tick / 100.0) * bq

            if not skip_sell and 1 <= ask_tick <= 99:
                if fv_skewed < ask_tick - min_cushion:
                    sq = qty
                    sc = (100 - ask_tick) / 100.0
                    cov = max(0.0, state.yes_inventory)
                    unc = max(0.0, sq - cov)
                    cost = sc * unc
                    if cost > available_cash * 0.14:
                        if sc > 0:
                            max_unc = (available_cash * 0.14) / sc
                            sq = min(sq, cov + max_unc)
                        sq = int(sq * 100) / 100.0
                    if sq >= 0.01:
                        actions.append(PlaceOrder(side=Side.SELL, price_ticks=ask_tick, quantity=sq))
                        unc_f = max(0.0, sq - max(0.0, state.yes_inventory))
                        available_cash -= sc * unc_f

        return actions
