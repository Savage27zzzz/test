from __future__ import annotations

from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import CancelAll, PlaceOrder, Side, StepState


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

        if state.steps_remaining < 30:
            return actions

        if comp_bid is None or comp_ask is None:
            return actions

        obs_spread = comp_ask - comp_bid
        if obs_spread < 4:
            return actions

        in_bid = comp_bid + 1
        in_ask = comp_ask - 1

        if in_bid >= in_ask or in_bid < 1 or in_ask > 99:
            return actions

        fv = self.fair_value
        net_inv = state.yes_inventory - state.no_inventory

        skew = -net_inv * 0.02
        skew = max(-2.0, min(2.0, skew))
        fv_skewed = fv + skew

        min_cushion = 1.0
        if obs_spread >= 8:
            min_cushion = 0.5

        place_bid = fv_skewed > in_bid + min_cushion
        place_ask = fv_skewed < in_ask - min_cushion

        if not place_bid and not place_ask:
            return actions

        if obs_spread >= 10:
            qty = 10.0
        elif obs_spread >= 8:
            qty = 8.0
        elif obs_spread >= 6:
            qty = 5.0
        else:
            qty = 3.0

        if state.steps_remaining < 200:
            qty = min(qty, 4.0)
        if state.steps_remaining < 80:
            qty = min(qty, 2.0)

        max_inv = 120.0
        skip_buy = net_inv > max_inv
        skip_sell = net_inv < -max_inv

        available_cash = state.cash

        if place_bid and not skip_buy:
            bq = qty
            cost = (in_bid / 100.0) * bq
            if cost > available_cash * 0.45:
                bq = (available_cash * 0.45) / (in_bid / 100.0)
            bq = int(bq * 100) / 100.0
            if bq >= 0.01:
                actions.append(PlaceOrder(side=Side.BUY, price_ticks=in_bid, quantity=bq))
                available_cash -= (in_bid / 100.0) * bq

        if place_ask and not skip_sell:
            sq = qty
            sc = (100 - in_ask) / 100.0
            cov = max(0.0, state.yes_inventory)
            unc = max(0.0, sq - cov)
            cost = sc * unc
            if cost > available_cash * 0.45:
                if sc > 0:
                    sq = min(sq, cov + (available_cash * 0.45) / sc)
            sq = int(sq * 100) / 100.0
            if sq >= 0.01:
                actions.append(PlaceOrder(side=Side.SELL, price_ticks=in_ask, quantity=sq))

        return actions
