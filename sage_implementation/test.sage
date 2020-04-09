execfile('f5_2.py');

var_count = 7
field_s = 65521
P = PolynomialRing(GF(field_s),var_count,'x');
I = sage.rings.ideal.Cyclic(P).homogenize();

print(P);
print('CPU\n=======================================================================')
gb = f5_2(I, use_cpu = True, field_size = field_s); f5_2.print_stats();

print(P);
print('GPU:\n=======================================================================')
gb = f5_2(I, field_size = field_s); f5_2.print_stats();