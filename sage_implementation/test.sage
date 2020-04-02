execfile('f5_2.py');
P = PolynomialRing(GF(32003),3,'x');
print(P);
I = sage.rings.ideal.Katsura(P).homogenize();
print('=======================================================================')
gb = f5_2(I); f5_2.print_stats();