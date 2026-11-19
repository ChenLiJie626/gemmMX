#!/bin/bash
# res.sh <prefix>  —— 从 .MetaXGPU.csdata 抠每个 kernel 的资源
S=/tmp/$1.s
awk '
  /^; Function info:/{inf=1}
  /^_Z[0-9A-Za-z_]+:/{ sym=$1; sub(/:.*/,"",sym) }
  /codeLenInByte/{len=$4}
  /NumMTRegsForWarps\/PEU/{mt=$3}
  /NumSTRegsForWarps\/PEU/{st=$3}
  /MaxWarpsPerPEU/{w=$3; if (mt != "") printf "  %-46s MTReg=%-5s STReg=%-5s warps/PEU=%-3s code=%s B\n", sym, mt, st, w, len; mt=""}
' $S
