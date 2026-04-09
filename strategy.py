from __future__ import annotations
import math
from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import CancelAll, PlaceOrder, Side, StepState

def _npdf(x): return math.exp(-0.5*x*x)/math.sqrt(2*math.pi)
def _icdf(p):
    if p<=1e-10: return -6.0
    if p>=1-1e-10: return 6.0
    if p<0.5: return -_icdf(1-p)
    t=math.sqrt(-2*math.log(1-p))
    return t-(2.515517+0.802853*t+0.010328*t*t)/(1+1.432788*t+0.189269*t*t+0.001308*t*t*t)
def _sticks(p,H):
    if H<=0 or p<=0.01 or p>=0.99: return 0.3
    v=H*0.0004
    if v<1e-15: return 0.3
    return max(0.2,2*_npdf(_icdf(p))/math.sqrt(v))

class Strategy(BaseStrategy):
    def __init__(self):
        self.fv=None; self.pb=None; self.pa=None

    def _ufv(self,cb,ca,st):
        if cb is not None and ca is not None:
            rm=(cb+ca)/2.0
            if self.fv is None: self.fv=rm
            else:
                if self.pa is not None and ca>self.pa:
                    self.fv=0.7*((self.pa+ca)/2)+0.3*self.fv
                elif self.pb is not None and cb<self.pb:
                    self.fv=0.7*((self.pb+cb)/2)+0.3*self.fv
                else: self.fv=0.6*rm+0.4*self.fv
        elif cb is not None: self.fv=cb+1.0
        elif ca is not None: self.fv=ca-1.0
        if self.fv is None: self.fv=50.0
        if st.buy_filled_quantity>0 and st.sell_filled_quantity==0: self.fv-=0.5
        elif st.sell_filled_quantity>0 and st.buy_filled_quantity==0: self.fv+=0.5
        self.fv=max(3,min(97,self.fv))
        self.pb,self.pa=cb,ca

    def on_step(self,state):
        a=[CancelAll()]
        cb=state.competitor_best_bid_ticks; ca=state.competitor_best_ask_ticks
        self._ufv(cb,ca,state)
        if state.steps_remaining<20 or cb is None or ca is None: return a
        obs=ca-cb
        if obs<3: return a
        fv=self.fv; ni=state.yes_inventory-state.no_inventory
        sk=max(-2.5,min(2.5,-ni*0.015))
        fvs=max(2,min(98,fv+sk))
        pf=max(0.02,min(0.98,fv/100))
        sig=_sticks(pf,state.steps_remaining)
        
        if sig < 0.55:
            ss=min(20.0, 6.0/max(0.2, sig))
            cm = 0.55
            bf = 0.22
            mi = 300
        elif sig < 0.75:
            ss=min(8.0, 3.0/max(0.3, sig))
            cm = 0.85
            bf = 0.16
            mi = 200
        else:
            ss=max(1.0, 2.0/max(0.5, sig))
            cm = 1.5
            bf = 0.12
            mi = 120

        sb=ni>mi; ss2=ni<-mi
        ac=state.cash
        ls=1.0
        if state.steps_remaining<100: ls=0.5
        if state.steps_remaining<40: ls=0.25
        bc=max(0.4,0.9*sig)*cm

        if obs>=8:
            levels=[(1,bc*0.5,25*ss),(2,bc*0.5+1,8*ss),(3,bc*0.5+2,4*ss)]
        elif obs>=6:
            levels=[(1,bc*0.7,14*ss),(2,bc*0.7+1,4*ss)]
        elif obs>=4:
            levels=[(1,bc,10*ss)]
        else:
            levels=[(1,bc*1.2,4*ss)]

        for off,mc,bq in levels:
            bt=cb+off; at=ca-off
            if bt>=at: continue
            q=max(0.01,int(bq*ls*100)/100)
            if not sb and 1<=bt<=99 and q>=0.01:
                if fvs>bt+mc:
                    b=q; c=(bt/100)*b
                    if c>ac*bf: b=int((ac*bf)/(bt/100)*100)/100
                    if b>=0.01: a.append(PlaceOrder(side=Side.BUY,price_ticks=bt,quantity=b)); ac-=(bt/100)*b
            if not ss2 and 1<=at<=99 and q>=0.01:
                if fvs<at-mc:
                    s=q; sc=(100-at)/100; cov=max(0,state.yes_inventory)
                    unc=max(0,s-cov); c=sc*unc
                    if c>ac*bf:
                        if sc>0: s=min(s,cov+(ac*bf)/sc)
                        s=int(s*100)/100
                    if s>=0.01: a.append(PlaceOrder(side=Side.SELL,price_ticks=at,quantity=s)); ac-=sc*max(0,s-max(0,state.yes_inventory))
        return a
