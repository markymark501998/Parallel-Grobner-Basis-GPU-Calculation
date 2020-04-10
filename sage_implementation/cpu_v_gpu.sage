execfile('f5_2.py');

var_count = 7
P = PolynomialRing(GF(32003),var_count,'x');
I = sage.rings.ideal.Cyclic(P).homogenize();

print(P);
print('CPU\n=======================================================================')
gb = f5_2(I, use_cpu = True); f5_2.print_stats();

print(P);
print('GPU:\n=======================================================================')
gb = f5_2(I); f5_2.print_stats();