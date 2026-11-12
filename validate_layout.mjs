// Host-only mathematical validation. Does NOT execute MACA or prove its ABI.
// Run: node validate_layout.mjs
import assert from 'node:assert/strict';

const output = [];
const index = (row, q, xor) => {
  const g = row >> 2;
  return 16*g + (((row & 3)*4+q) ^ (xor ? g & 15 : 0));
};
const inputA = (m,k) => ((m*7+k*3+11)%13)-6;
const inputB = (n,k) => ((n*5+k*7+3)%17)-8;

for (const BM of [128]) for (const xor of [false,true]) {
  // Emulate all global-to-BSM producers, using the wave-uniform-base ABI.
  const sa = new Array(BM*4), sb = new Array(BM*4);
  const seen = new Set();
  const threads = (BM/64)**2*64;
  for (let h=0;h<BM*4/threads;h++) for (let tid=0;tid<threads;tid++) {
    const wave = tid>>6, lane = tid&63;
    const physical = h*threads+wave*64+lane;
    const group=physical>>4, col=(physical&15)^(xor?group&15:0);
    const row=group*4+(col>>2), q=col&3;
    assert.equal(index(row,q,xor),physical);
    assert(!seen.has(row*4+q)); seen.add(row*4+q);
    sa[physical]=Array.from({length:8},(_,z)=>inputA(row,8*q+z));
    sb[physical]=Array.from({length:8},(_,z)=>inputB(row,8*q+z));
  }
  assert.equal(seen.size,BM*4);
  const written=new Set();
  for(let wm=0;wm<BM/64;wm++) for(let wn=0;wn<BM/64;wn++) {
    const fa=Array.from({length:64},(_,lane)=>Array.from({length:4},(_,i)=>
      sa[index(wm*64+4*(lane&15)+i,lane>>4,xor)]));
    const fb=Array.from({length:64},(_,lane)=>Array.from({length:4},(_,j)=>
      sb[index(wn*64+4*(lane&15)+j,lane>>4,xor)]));
    // A native 16x16x16 instruction distributes input K over four quarters.
    // First operand's output coordinate is 4*quarter+component, second's is r.
    // This contract is inferred from the public native kernel's store mapping.
    for(let lane=0;lane<64;lane++) for(let i=0;i<4;i++)
    for(let j=0;j<4;j++) for(let z=0;z<4;z++) {
      const secondRow=lane&15, firstRow=4*(lane>>4)+z;
      let actual=0;
      for(let half=0;half<2;half++) for(let q=0;q<4;q++) for(let e=0;e<4;e++)
        actual+=fa[16*q+secondRow][i][half*4+e]*fb[16*q+firstRow][j][half*4+e];
      const m=wm*64+secondRow*4+i, n=wn*64+firstRow*4+j;
      let expected=0;
      for(let k=0;k<32;k++) expected+=inputA(m,k)*inputB(n,k);
      assert.equal(actual,expected,`BM=${BM} XOR=${xor} (${m},${n})`);
      const ix=m*BM+n;
      assert(!written.has(ix)); written.add(ix);
    }
  }
  assert.equal(written.size,BM*BM);
  output.push(`PASS: BM=${BM}, XOR=${xor}, producer bijection, native MMA model, unique stores`);
}

// Circular-buffer ownership: every read must see its own K slice; writes must
// not replace a slice before all consumers release it. Counters are structural
// checks only: the hardware's completion/visibility semantics require a GPU.
for(const tiles of [8,16,24,32]) for(const stages of [1,2,3,4]) {
  const ring=new Array(stages).fill(null), released=new Set();
  const issue=t=>{
    assert(t<tiles);
    const old=ring[t%stages];
    assert(old===null||released.has(old),`overwrite ${old} before release`);
    ring[t%stages]=t;
  };
  for(let t=0;t<stages-1;t++) issue(t);
  for(let t=0;t<tiles;t++) {
    if(t+stages-1<tiles) issue(t+stages-1);
    assert.equal(ring[t%stages],t);
    const newer=Math.min(tiles-t-1,stages-1);
    assert(newer>=0&&newer<=3);
    released.add(t);
  }
  if(stages>=2) {
    ring.fill(null); released.clear();
    for(let t=0;t<stages;t++) issue(t);
    let current=0;
    for(let t=0;t<tiles;t++) {
      assert.equal(current,t);
      if(t+1<tiles) assert.equal(ring[(t+1)%stages],t+1);
      released.add(t);
      if(t+stages<tiles) issue(t+stages);
      current=t+1;
    }
  }
}
output.push('PASS: ring ownership and prologue/drain, stages=1..4, K=256/512/768/1024');
console.log(output.join('\n'));
console.log('NOT TESTED: mxcc compilation, builtin ABI, GPU visibility, numerical accuracy, performance.');
