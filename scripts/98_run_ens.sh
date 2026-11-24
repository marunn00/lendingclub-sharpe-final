#!/bin/bash
# 98_c 단계 체인 (채택 기법 = 98_b 스윕 결과: 고유값 빈도+TE, 주/목적 TE, 발행연도, 신용 파생, spread_fedfunds, 국채 참고 2열)
cd "$(dirname "$0")/.."; W=산출물/앙상블_v1/_work
K=T3_hc_both,T3_state_te,T7_year,T4_credit,T4_spread_ff,T4_tsy_ref
for s in kept tune fit ens prune; do python3 scripts/98_c_ens.py --stage $s --kept $K --trials 25 >> $W/c.log 2>&1; done
echo CHAIN_DONE >> $W/c.log
