execfile('f5_2.py');

var_count = 4
P = PolynomialRing(GF(65521),var_count,'x');
I = sage.rings.ideal.Cyclic(P).homogenize();

print(P);
print('CPU\n=======================================================================')
gb = f5_2(I, use_cpu = True, field_size = 65521); f5_2.print_stats();

print(P);
print('GPU:\n=======================================================================')
gb = f5_2(I, field_size = 65521, print_debug = True); f5_2.print_stats();