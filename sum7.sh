#!/bin/bash
cd /root/gemm
blk() {  # blk <prefix> <full symbol>
  S=/tmp/$1.s; SYM=$2
  awk "/^$SYM:/,/End function/" $S | grep -vE '^\s*\.|^\s*$' > /tmp/kk.s
  awk -v tag="$1" -v sym="$SYM" '
   /^(LBB[0-9_]+:|; %bb\.[0-9]+:)/{ sub(/:.*/,"",$0); blk=$0; gsub(/^; /,"",blk); order[++n]=blk }
   /^[ \t]+[a-z]/{tot[blk]++}
   /[ \t]mma_i32|[ \t]mma_16x16/{mma[blk]++}
   /[ \t]lds_b/{lds[blk]++}
   /[ \t]ldg_b128/{ldg[blk]++}
   /[ \t]sts_b/{sts[blk]++}
   /[ \t]stg_b/{stg[blk]++}
   /[ \t]arrive/{arr[blk]++}
   /[ \t]snop/{snop[blk]++; snsum[blk]+=$2+0}
   /[ \t]barrier/{bar[blk]++}
   END{
     best=""; bm=-1
     for(i=1;i<=n;i++){b=order[i]; if(mma[b]+0>bm){bm=mma[b]+0;best=b}}
     b=best
     printf "%-10s %-44s %-9s 总=%-4d mma=%-4d(%5.1f%%) lds=%-3d ldg=%-3d sts=%-3d arrive=%-3d snop=%-3d(%-3d拍) bar=%d\n",
       tag, sym, b, tot[b]+0, mma[b]+0, 100*(mma[b]+0)/(tot[b]+0.0001), lds[b]+0, ldg[b]+0, sts[b]+0, arr[b]+0, snop[b]+0, snsum[b]+0, bar[b]+0
   }' /tmp/kk.s
}
echo "=== 主循环块（mma 最多的那个块）==="
blk a_base  _Z7mma_256ILb1ELb1EEvPKaS1_Piiii
blk a_i0    _Z7mma_256ILb1ELb1EEvPKaS1_Piiii
blk a_i4    _Z7mma_256ILb1ELb1EEvPKaS1_Piiii
blk a_g0    _Z7mma_256ILb1ELb1EEvPKaS1_Piiii
blk a_pred  _Z7mma_256ILb1ELb1EEvPKaS1_Piiii
blk a_c600s _Z8mma_tileILb1ELb1EEvPKaS1_Piiii
blk a_c600s _Z8mma_tileILb0ELb1EEvPKaS1_Piiii
blk a_c500  _Z7mma_256ILb1ELb1EEvPKaS1_Piiii
blk a_c500  _Z7mma_256ILb1ELb0EEvPKaS1_Piiii
blk a_pipe  _Z8mma_pipeILi256ELi0EEvPKaS1_Piiii
blk a_pipe  _Z8mma_pipeILi256ELi1EEvPKaS1_Piiii
blk a_pipe  _Z8mma_pipeILi128ELi1EEvPKaS1_Piiii
blk a_pipe  _Z8mma_pipeILi128ELi2EEvPKaS1_Piiii
blk a_dbuf  _Z8mma_dbufILi64ELi3ELb0ELb1EEvPKaS1_Piiii
