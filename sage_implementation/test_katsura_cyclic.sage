execfile('f5_2.py');
print('Katsura');
var_count = 5
P = PolynomialRing(GF(65521),var_count,'x');
I = sage.rings.ideal.Katsura(P).homogenize();

print(P);
print('CPU\n=======================================================================')
gb = f5_2(I, use_cpu = True, field_size = 65521); f5_2.print_stats();

print(P);
print('GPU:\n=======================================================================')
gb = f5_2(I, field_size = 65521); f5_2.print_stats();

var_count = 6
P = PolynomialRing(GF(65521),var_count,'x');
I = sage.rings.ideal.Katsura(P).homogenize();

print(P);
print('CPU\n=======================================================================')
gb = f5_2(I, use_cpu = True, field_size = 65521); f5_2.print_stats();

print(P);
print('GPU:\n=======================================================================')
gb = f5_2(I, field_size = 65521); f5_2.print_stats();

var_count = 7
P = PolynomialRing(GF(65521),var_count,'x');
I = sage.rings.ideal.Katsura(P).homogenize();

print(P);
print('CPU\n=======================================================================')
gb = f5_2(I, use_cpu = True, field_size = 65521); f5_2.print_stats();

print(P);
print('GPU:\n=======================================================================')
gb = f5_2(I, field_size = 65521); f5_2.print_stats();

var_count = 8
P = PolynomialRing(GF(65521),var_count,'x');
I = sage.rings.ideal.Katsura(P).homogenize();

print(P);
print('CPU\n=======================================================================')
gb = f5_2(I, use_cpu = True, field_size = 65521); f5_2.print_stats();

print(P);
print('GPU:\n=======================================================================')
gb = f5_2(I, field_size = 65521); f5_2.print_stats();

var_count = 9
P = PolynomialRing(GF(65521),var_count,'x');
I = sage.rings.ideal.Katsura(P).homogenize();

print(P);
print('CPU\n=======================================================================')
gb = f5_2(I, use_cpu = True, field_size = 65521); f5_2.print_stats();

print(P);
print('GPU:\n=======================================================================')
gb = f5_2(I, field_size = 65521); f5_2.print_stats();



print('Cyclic');
var_count = 5
P = PolynomialRing(GF(65521),var_count,'x');
I = sage.rings.ideal.Cyclic(P).homogenize();

print(P);
print('CPU\n=======================================================================')
gb = f5_2(I, use_cpu = True, field_size = 65521); f5_2.print_stats();

print(P);
print('GPU:\n=======================================================================')
gb = f5_2(I, field_size = 65521); f5_2.print_stats();

var_count = 6
P = PolynomialRing(GF(65521),var_count,'x');
I = sage.rings.ideal.Cyclic(P).homogenize();

print(P);
print('CPU\n=======================================================================')
gb = f5_2(I, use_cpu = True, field_size = 65521); f5_2.print_stats();

print(P);
print('GPU:\n=======================================================================')
gb = f5_2(I, field_size = 65521); f5_2.print_stats();

var_count = 7
P = PolynomialRing(GF(65521),var_count,'x');
I = sage.rings.ideal.Cyclic(P).homogenize();

print(P);
print('CPU\n=======================================================================')
gb = f5_2(I, use_cpu = True, field_size = 65521); f5_2.print_stats();

print(P);
print('GPU:\n=======================================================================')
gb = f5_2(I, field_size = 65521); f5_2.print_stats();

var_count = 8
P = PolynomialRing(GF(65521),var_count,'x');
I = sage.rings.ideal.Cyclic(P).homogenize();

print(P);
print('CPU\n=======================================================================')
gb = f5_2(I, use_cpu = True, field_size = 65521); f5_2.print_stats();

print(P);
print('GPU:\n=======================================================================')
gb = f5_2(I, field_size = 65521); f5_2.print_stats();

var_count = 9
P = PolynomialRing(GF(65521),var_count,'x');
I = sage.rings.ideal.Cyclic(P).homogenize();

print(P);
print('CPU\n=======================================================================')
gb = f5_2(I, use_cpu = True, field_size = 65521); f5_2.print_stats();

print(P);
print('GPU:\n=======================================================================')
gb = f5_2(I, field_size = 65521); f5_2.print_stats();