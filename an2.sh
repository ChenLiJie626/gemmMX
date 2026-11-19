#!/bin/bash
# an2.sh <prefix> <sym-regex>   —— 真汇编分析
S=/tmp/$1.s
SYM=$(grep -oE "^_Z[0-9A-Za-z_]*$2[A-Za-z0-9_]*" $S | head -1)
[ -z "$SYM" ] && { echo "找不到符号 $2"; exit 1; }
echo "########## $1 / $SYM"
awk "/^$SYM:/,/End function/" $S | grep -vE '^\s*\.|^\s*$' > /tmp/k_$1.s
echo "  函数指令行: $(grep -cE '^\s+[a-z]' /tmp/k_$1.s)"
# 资源
awk "/^; Function info:/,0" $S > /tmp/info_$1.txt
grep -A6 -B1 "$SYM" /tmp/info_$1.txt | head -8
sed -n "/codeLenInByte/,+3p" $S | head -4
echo "  --- 全函数助记符 top12 ---"
grep -oE '^\s+[a-z][a-z0-9_.]*' /tmp/k_$1.s | tr -d ' \t' | sort | uniq -c | sort -rn | head -12 | sed 's/^/   /'
echo "  --- 每个基本块 ---"
awk '
 /^(LBB[0-9_]+:|; %bb\.[0-9]+:)/{ sub(/:.*/,"",$0); blk=$0; gsub(/^; /,"",blk); order[++n]=blk }
 /[ \t]mma_i32|[ \t]mma_16x16/{mma[blk]++}
 /[ \t]lds_b/{lds[blk]++}
 /[ \t]ldg_b128/{ldg[blk]++}
 /[ \t]stg_b/{stg[blk]++}
 /[ \t]sts_b/{sts[blk]++}
 /[ \t]arrive/{arr[blk]++}
 /[ \t]snop/{snop[blk]++; sn=$2+0; snsum[blk]+=sn}
 /[ \t]barrier/{bar[blk]++}
 END{for(i=1;i<=n;i++){b=order[i];
   printf "   %-14s mma=%-4d lds=%-3d ldg=%-3d sts=%-3d stg=%-4d arrive=%-3d snop=%-3d(共%-4d拍) barrier=%d\n",
     b, mma[b]+0, lds[b]+0, ldg[b]+0, sts[b]+0, stg[b]+0, arr[b]+0, snop[b]+0, snsum[b]+0, bar[b]+0}}' /tmp/k_$1.s
# 主循环 = mma 最多且有 ldg 的块
BLK=$(awk '/^(LBB[0-9_]+:|; %bb\.[0-9]+:)/{sub(/:.*/,"",$0); blk=$0; gsub(/^; /,"",blk)} /[ \t]mma_i32|[ \t]mma_16x16/{mm[blk]++} /[ \t]ldg_b128/{g[blk]++} END{m=-1;for(b in g) if(mm[b]+0>0&&g[b]+0>m){m=g[b]+0;bb=b}; if(bb==""){m=-1;for(b in mm) if(mm[b]+0>m){m=mm[b]+0;bb=b}} print bb}' /tmp/k_$1.s)
echo "  --- 主循环块 $BLK 的发射时间线 ---"
awk -v b="$BLK" '
  $0 ~ "^"b":" {f=1;next}
  f && /^(LBB[0-9_]+:|; %bb\.[0-9]+:)/{exit}
  f{print}' /tmp/k_$1.s > /tmp/blk_$1.s
awk '
  /[ \t]mma_i32|[ \t]mma_16x16/{m++; next}
  /[ \t]ldg_b128/{printf "   mma#%-4d %s\n", m, $0; next}
  /[ \t]lds_b/{l[m]++; next}
  /[ \t]arrive/{printf "   mma#%-4d %s\n", m, $0; next}
  /[ \t]barrier/{printf "   mma#%-4d BARRIER\n", m; next}
  END{printf "   lds 读分布(mma#): "; for(k in l) printf "%d(x%d) ", k, l[k]; printf "\n"}
' /tmp/blk_$1.s | sed 's/\t/ /g'
echo "  --- 主循环块计数 ---"
TOT=$(grep -cE '^\s+[a-z]' /tmp/blk_$1.s)
MMA=$(grep -cE '[ \t]mma_i32|[ \t]mma_16x16' /tmp/blk_$1.s)
SNSUM=$(awk '/[ \t]snop/{s+=$2+0} END{print s+0}' /tmp/blk_$1.s)
printf "   总指令=%s  mma=%s (%.1f%%)  snop 总拍数=%s\n" "$TOT" "$MMA" "$(awk -v a=$MMA -v b=$TOT 'BEGIN{print 100*a/b}')" "$SNSUM"
