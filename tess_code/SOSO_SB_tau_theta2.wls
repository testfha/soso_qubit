#!/usr/bin/env wolframscript
(* ::Package:: *)

ClearAll["Global`*"];

$Assumptions =
  Element[{g1, g2, g3, theta1, n1, phi1, lam, dJ}, Reals] &&
  g1 > g3 > 0 && g2 > g3 && 0 <= n1 <= 1;

id = IdentityMatrix[2];
s = PauliMatrix[#]/2 & /@ Range[3];

op[m_, k_] := KroneckerProduct @@ ReplacePart[ConstantArray[id, 3], k -> m];
S[k_] := op[#, k] & /@ s;
dot[a_, b_] := Total[MapThread[Dot, {a, b}]];
axis[n_, ph_] := {n Sin[ph], n Cos[ph], Sqrt[1 - n^2]};
rot[v_, n_, ph_, th_] := RotationMatrix[th, axis[n, ph]].v;

Hexact[J_, th_] :=
  g1 S[1][[3]] + g2 S[2][[3]] + g3 S[3][[3]] +
  J dot[S[1], rot[S[2], n1, phi1, th]];

J0 = 2 (g1 - g3) (g2 - g3)/(g1 + g2 - 2 g3);
w[J_] := Sqrt[(g1 - g2)^2 + J^2];
tau[J_] := ArcTan[g1 - g2, J];

U12[J_] := {
  {1, 0, 0, 0},
  {0, Cos[tau[J]/2], -Sin[tau[J]/2], 0},
  {0, Sin[tau[J]/2],  Cos[tau[J]/2], 0},
  {0, 0, 0, 1}
};

toTau[M_, J_] := Simplify[
  ConjugateTranspose[KroneckerProduct[U12[J], id]].M.KroneckerProduct[U12[J], id],
  $Assumptions
];

U0 = KroneckerProduct[U12[J0], id];
toTau0[M_] := Simplify[ConjugateTranspose[U0].M.U0, $Assumptions];

h0[J_] := h0[J] = Simplify[toTau[Hexact[J, 0], J], $Assumptions];

hso[J_] := hso[J] = Simplify[
  Expand @ Normal @ Series[toTau[Hexact[J, lam theta1], J] - h0[J], {lam, 0, 2}],
  $Assumptions
];

(* In the fixed idle basis U0 = U12[J0] \[CircleTimes] I, the idle crossing is
   between the lower tau-mixed mz=-1/2 state and T_- up. *)
logIdx = {6, 7};
Qidx = Complement[Range[8], logIdx];
Pblock[M_] := M[[logIdx, logIdx]];

Bdiag[M_] := Module[{B = ConstantArray[0, Dimensions[M]]},
  B[[logIdx, logIdx]] = M[[logIdx, logIdx]];
  B[[Qidx, Qidx]] = M[[Qidx, Qidx]];
  B
];

Sgen[H0_, H1_] := Module[{e = Diagonal[H0], V = H1 - Bdiag[H1], inP},
  inP[i_] := MemberQ[logIdx, i];
  Table[
    If[Xor[inP[i], inP[j]], V[[i, j]]/(e[[i]] - e[[j]]), 0],
    {i, 8}, {j, 8}
  ]
];

h0IdleShift = Simplify[
  Expand @ Normal @ Series[toTau0[Hexact[J0 (1 + dJ lam^2), 0]], {lam, 0, 2}],
  $Assumptions
];

hsoIdle = Simplify[
  Expand @ Normal @ Series[toTau0[Hexact[J0, lam theta1] - Hexact[J0, 0]], {lam, 0, 2}],
  $Assumptions
];

series = h0IdleShift + hsoIdle;

A0 = Coefficient[series, lam, 0];
A1 = Coefficient[series, lam, 1];
A2 = Coefficient[series, lam, 2];
G1 = Sgen[A0, A1];

Heff = Expand[
  Pblock[A0] +
  lam Pblock[A1] +
  lam^2 Pblock[A2 + 1/2 (G1.A1 - A1.G1)]
];

Heff2 = Coefficient[Heff, lam, 2];
split2 = Simplify[Heff2[[1, 1]] - Heff2[[2, 2]], $Assumptions];

dJRule = First @ Solve[split2 == 0, dJ];
dJFromSW = Simplify[dJ /. dJRule, $Assumptions];

dJStefano = Simplify[
  g1 g2 (
    g1^3 + g1^2 (g2 - 3 g3) + (g2 - g3)^3 + g1 (g2^2 - 3 g2 g3 + 3 g3^2)
  )/(2 (g1 + g2 - 2 g3) (g1 + g2 - g3) g3 (g1^2 + g2^2 - (g1 + g2) g3))
  * n1^2 theta1^2,
  $Assumptions
];

checkStefano = Simplify[dJFromSW - dJStefano, $Assumptions];

results = <|
  "J0" -> J0,
  "tau0" -> tau[J0],
  "logIdx" -> logIdx,
  "dJFromSW" -> dJFromSW,
  "dJStefano" -> dJStefano,
  "checkStefano" -> checkStefano
|>;

results
