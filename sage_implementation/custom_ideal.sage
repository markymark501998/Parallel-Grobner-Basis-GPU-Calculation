execfile('f5_2.py');

R.<x,y,z,k> = GF(65521)[]
I = R.ideal([4 + 3*x + x^2, 1 + y^2, 4 + z^3, 2*k*x]).homogenize()

print(R);
print('GPU:\n=======================================================================')
gb = f5_2(I, field_size = 65521); f5_2.print_stats();