# -*- coding: utf-8 -*-
"""
Jean-Charles Faugère's F5 Algorithm in F4-Style.

This variant of F5 proceed degree-by-degree in the outer loop instead
of by index of the generators. Futhermore, this variant uses linear
algebra to perform the top reductions.

AUTHORS:
- Martin Albrecht and John Perry (2009-01): use linear algebra to
  perform reductions in F5 proper
- Martin Albrecht (2009-04): proceed degree by degree, performance
  improvements, clean-ups, documentation

EXAMPLE::

    sage: execfile('f5_2.py')
    sage: P = PolynomialRing(GF(32003),3,'x')
    sage: I = sage.rings.ideal.Katsura(P).homogenize()
    sage: gb = f5_2(I); f5_2.print_stats()
     3    1    5 x    8,    5,    0
                   |L|:    4
               L is GB: True
    reductions to zero:    0
           max. degree:    3

    sage: I = sage.rings.ideal.Cyclic(P).homogenize()
    sage: gb = f5_2(I); f5_2.print_stats()
     0    0
                   |L|:    3
               L is GB: True
    reductions to zero:    0
           max. degree:    3

    sage: P = PolynomialRing(GF(32003),4,'x')
    sage: I = sage.rings.ideal.Katsura(P).homogenize()
    sage: gb = f5_2(I); f5_2.print_stats()
     3    2   10 x   17,   10,    0
     4    1   10 x   17,   10,    0
                   |L|:    8
               L is GB: True
    reductions to zero:    0
           max. degree:    4

    sage: I = sage.rings.ideal.Cyclic(P).homogenize()
    sage: gb = f5_2(I); f5_2.print_stats()
     4    1    4 x    6,    3,    1
     5    2    6 x   12,    6,    0
     6    1    3 x    9,    3,    0
                   |L|:    8
               L is GB: True
    reductions to zero:    1
           max. degree:    6

    sage: P = PolynomialRing(GF(32003),5,'x')
    sage: I = sage.rings.ideal.Katsura(P).homogenize()
    sage: gb = f5_2(I); f5_2.print_stats()
     3    4   18 x   32,   18,    0
     4    3   28 x   43,   28,    0
     5    1   24 x   39,   24,    0
                   |L|:   15
               L is GB: True
    reductions to zero:    0
           max. degree:    5

    sage: I = sage.rings.ideal.Cyclic(P).homogenize()
    sage: gb = f5_2(I); f5_2.print_stats()
     4    1    8 x   24,    8,    0
     5    2   15 x   33,   15,    0
     6    6   22 x   41,   22,    0
     7    6   22 x   40,   22,    0
     8    8   18 x   36,   18,    0
     9   10   19 x   37,   19,    0
    10    4
}
    reductions to zero:    0
           max. degree:   13

    sage: P = PolynomialRing(GF(32003),6,'x')
    sage: I = sage.rings.ideal.Katsura(P).homogenize()
    sage: gb = f5_2(I); f5_2.print_stats()
     3    5   27 x   52,   27,    0
     4    6   56 x   86,   56,    0
     5    4   80 x  111,   80,    0
     6    2   57 x   88,   57,    0
                   |L|:   30
               L is GB: True
    reductions to zero:    0
           max. degree:    6

    sage: I = sage.rings.ideal.Cyclic(P).homogenize()
    sage: gb = f5_2(I); f5_2.print_stats()
     4    1    9 x   40,    9,    0
     5    2   17 x   59,   17,    0
     6    6   42 x  100,   42,    0
     7   15   55 x  118,   55,    0
     8   21   76 x  137,   75,    1
     9   29  150 x  210,  148,    2
    10   35  130 x  188,  124,    6
    11   25   87 x  145,   85,    2
    12   16   12 x   38,   12,    0
    13    7   32 x   58,   32,    0
    14   13   41 x   67,   41,    0
    15   23   35 x   61,   35,    0
    16    4   18 x   44,   18,    0
    17    2   36 x   62,   36,    0
                   |L|:  172
               L is GB: True
    reductions to zero:   11
           max. degree:   17

    sage: P = PolynomialRing(GF(7),3,'x')
    sage: I = sage.rings.ideal.Katsura(P).homogenize()
    sage: gb = f5_2(I); f5_2.print_stats()
     3    1    5 x    8,    5,    0
                   |L|:    4
               L is GB: True
    reductions to zero:    0
           max. degree:    3

    sage: I = sage.rings.ideal.Cyclic(P).homogenize()
    sage: gb = f5_2(I); f5_2.print_stats()
     0    0
                   |L|:    3
               L is GB: True
    reductions to zero:    0
           max. degree:    3

    sage: P = PolynomialRing(GF(7),4,'x')
    sage: I = sage.rings.ideal.Katsura(P).homogenize()
    sage: gb = f5_2(I); f5_2.print_stats()
     3    2   11 x   18,   11,    0
     4    2   17 x   23,   17,    0
                   |L|:    8
               L is GB: True
    reductions to zero:    0
           max. degree:    4

    sage: I = sage.rings.ideal.Cyclic(P).homogenize()
    sage: gb = f5_2(I); f5_2.print_stats()
     4    1    4 x    6,    3,    1
     5    2    6 x   12,    6,    0
     6    1    3 x    9,    3,    0
                   |L|:    8
               L is GB: True
    reductions to zero:    1
           max. degree:    6

    sage: P = PolynomialRing(GF(7),5,'x')
    sage: I = sage.rings.ideal.Katsura(P).homogenize()
    sage: gb = f5_2(I); f5_2.print_stats()
     3    4   18 x   32,   18,    0
     4    3   28 x   43,   28,    0
     5    1   23 x   38,   23,    0
                   |L|:   15
               L is GB: True
    reductions to zero:    0
           max. degree:    5

    sage: I = sage.rings.ideal.Cyclic(P).homogenize()
    sage: gb = f5_2(I); f5_2.print_stats()
     4    1    8 x   24,    8,    0
     5    2   15 x   33,   15,    0
     6    6   22 x   41,   22,    0
     7    6   22 x   40,   22,    0
     8    7   17 x   35,   17,    0
     9   10   19 x   37,   19,    0
    10    3
    11    6    8 x   22,    8,    0
    12    3    9 x   23,    9,    0
    13    3    8 x   22,    8,    0
                   |L|:   39
               L is GB: True
    reductions to zero:    0
           max. degree:   13

    sage: P = PolynomialRing(GF(7),6,'x')
    sage: I = sage.rings.ideal.Katsura(P).homogenize()
    sage: gb = f5_2(I); f5_2.print_stats()
     3    5   27 x   52,   27,    0
     4    6   55 x   85,   55,    0
     5    4   62 x   92,   62,    0
     6    6   81 x  110,   81,    0
     7    1   86 x  115,   86,    0
     8    1
                   |L|:   39
               L is GB: True
    reductions to zero:    0
           max. degree:    7

    sage: I = sage.rings.ideal.Cyclic(P).homogenize()
    sage: gb = f5_2(I); f5_2.print_stats()
     4    1    9 x   40,    9,    0
     5    2   17 x   59,   17,    0
     6    6   42 x  100,   42,    0
     7   15   55 x  118,   55,    0
     8   21   76 x  137,   75,    1
     9   29  148 x  208,  146,    2
    10   36  129 x  187,  123,    6
    11   29  102 x  161,  100,    2
    12   31   47 x  101,   47,    0
    13   11   32 x   58,   32,    0
    14   19   43 x   69,   43,    0
    15   28   36 x   62,   36,    0
    16    4   18 x   44,   18,    0
    17    7   40 x   66,   40,    0
    18    1
                   |L|:  179
               L is GB: True
    reductions to zero:   11
           max. degree:   17

    sage: sr = mq.SR(1,1,1,4)
    sage: F,s = sr.polynomial_system()
    sage: I = Ideal(F.gens()).homogenize()
    sage: gb = f5_2(I); f5_2.print_stats()
     3   22   64 x   53,   49,   15
     4   33   80 x   51,   49,   31
     5   14   28 x   23,   21,    7
                   |L|:   88
               L is GB: True
    reductions to zero:   53
           max. degree:    5
"""

from itertools import chain
import time
import cython
import gpuadder

elim_run_time = 0.0
sym_run_time = 0.0
total_run_time = 0.0

field_size_var = 32003
print_debug_var = False
use_cpu_var = False

def idx2c(i,j,ld):
    return ((j*ld)+(i))

def compare_by_degree(f,g):
    """
    Compare ``f`` and ``g`` with respect to their degree first and
    only if those match w.r.t. the monomial ordering.

    INPUT:
    
    - ``f`` - a polynomial
    - ``g`` - a polynomial
    """
    if f.total_degree() > g.total_degree():
        return 1
    elif f.total_degree() < g.total_degree():
        return -1
    else:
        return cmp(f, g)

class F5_2:
    """
    Jean-Charles Faugère's F5 Algorithm in F4-Style.

    This variant of F5 proceed degree-by-degree in the outer loop
    instead of by index of the generators. Futhermore, this variant
    uses linear algebra to perform the top reductions.
    """
    def __init__(self):
        self.L = [] # labels
        self.Rules = [] # rewriting rules
        self.verbose = 0
        self.zero_reductions = 0 # we count reductions to zero
        
    def __call__(self, F, field_size = 32003, print_debug=False, use_cpu=False, D=None, proof=False):
        """
        Compute a Gröbner basis for the input system ``F``. 

        If ``D`` is not ``None`` a D-Gröbner Basis is computed
        instead. If ``proof`` is ``True`` we invoke the Buchberger
        criterion whether the given basis is a Gröbner basis to force
        termination. However, we expect F5 to terminate on all inputs
        and thus ``proof`` defaults to ``False`` for now.

        INPUT:

        - ``F`` - a list of polynomials
        - ``D`` - a maximal degree (default: ``None``)
        - ``proof`` - whether to provably force termination (default: ``False``)
        """
        self.__init__()

        L, Rules = self.L, self.Rules
        critical_pair = self.critical_pair
        select = self.select
        compute_spols = self.compute_spols
        sig, poly = self.sig, self.poly
        is_top_reducible = self.is_top_reducible

        global elim_run_time
        global sym_run_time
        global total_run_time

        elim_run_time = 0
        sym_run_time = 0
        total_run_time = 0

        global field_size_var
        global print_debug_var
        global use_cpu_var

        field_size_var = field_size
        print_debug_var = print_debug
        use_cpu_var = use_cpu

        tt_start = time.clock()

        try:
            F = F.gens()
        except AttributeError:
            pass

        F = Ideal(F).interreduced_basis()
        F = sorted(F, cmp=compare_by_degree)
        m = len(F)

        assert(all(f == f.homogenize() for f in F))

        R = F[0].parent()

        # First, we encode that each f_i is generated by itself
        P,G = set(), []
        for i,f in enumerate(F):
            L.append( (Signature(R(1),i), f*f.lc()**(-1)) )
            Rules.append([])
            self.add_rule(Signature(R(1),i),i)
            P = P.union(set([critical_pair(i, g, G) for g in chain(*G)]))
            G.append([i])

        d = 0
        while P: # as long as we have S-polynomials
            iteration_s_time = time.clock()
            # TODO: this is a technicality, get rid of this
            P = set([p for p in P if p != tuple()])

            d = self.minimal_d(P) # get the minimal d
            Pd = select(P,d) # and select all pairs with minimal degree
            print "%2d %4d"%(d, len(Pd)),

            P = P.difference(Pd)

            # we allow a degree bound D
            if D is not None and d > D:
                print
                continue
            
            # compute the S-polynomials

            S = compute_spols(Pd)

            if len(S):
                self.degree_reached = d

            # and perform top-reductions
            Stilde = self.reduction(S, G)

            # we add each new polynomial, and repeat
            for h in Stilde:
                Pnew = set([critical_pair(h, g, G) for g in chain(*G)])
                P = P.union(Pnew)
                G[sig(h)[1]].append(h)
                
            P = set([p for p in P if p != tuple()])

            sys.stdout.flush()
            if proof and self.terminate(P, G):
                iteration_e_time = time.clock()
                iter_time = iteration_e_time - iteration_s_time
                print (' Iteration Time: [%6f]')%iter_time,
                print
                break

            iteration_e_time = time.clock()
            iter_time = iteration_e_time - iteration_s_time
            print (' Iteration Time: [%6f]')%iter_time,
            print
        
        tt_end = time.clock()
        total_run_time = tt_end - tt_start

        return [poly(f) for f in chain(*G)]

    def critical_pair(self, k, l, G):
        """
        Return the critical pair for the polynomials ``k`` and ``l``
        indexed in ``L`` iff the F5 criteria pass.

        Otherwise, return the empty tuple.

        INPUT:
        
        - ``k`` - a polynomial index for ``L``
        - ``l`` - a polynomial index for ``L``
        - ``G`` - the intermediate Gröbner basis indexed in ``L``

        adapted from Justin Gash (p.51): 
       
        'It is the subroutine critical_pair that is responsible for
         imposing the F5 Criterion from Theorem 3.3.1. Note that in
         condition (3) of Theorem 3.3.1, it is required that all pairs
         (r_i, r_j) be normalized. The reader will recall from
         Definition 3.2.2 that a pair is normalized if: 
         
         (1) S(k) = m_k*F_{e_k} is not top-reducible by <f_0, ..., f_{e_k}-1> 

         (2) S(l) = m_l*F_{e_l} is not top-reducible by <f_0, ..., f_{e_l}-1>
         
         (3) S(m_k*k) > S(m_l*l)

         If these three conditions are not met in ``critical_pair()``
         (note that the third condition will always be met because
         ``cirtical_pair()`` forces it to be met), the nominated
         critical pair is dropped and () is returned.

         Once we have collected the nominated critical pairs that pass
         the F5 criterion test of ``critical_pair(()``, we send them
         to ``compute_spols()``.'
        """
        poly = self.poly
        sig = self.sig
        is_top_reducible = self.is_top_reducible
        is_rewritable = self.is_rewritable
        LCM = lambda f,g: f.parent().monomial_lcm(f,g)

        tk = poly(k).lt()
        tl = poly(l).lt()
        t = LCM(tk, tl)
        uk = t//tk
        ul = t//tl
        mk, ek = sig(k)
        ml, el = sig(l)

        # they are are same
        if ek == el and uk*mk == ul*ml:
            return tuple()
        
        if is_top_reducible(uk*mk, G[:ek]):
            return tuple()

        if is_top_reducible(ul*ml, G[:el]):
            return tuple()

        # this check is in compute_spols() again, but we can filter
        # some stuff out here already
        if is_rewritable(uk, k) or is_rewritable(ul, l):
            return tuple()

        # preserve order
        if uk * sig(k) < ul * sig(l):
            uk, ul = ul, uk
            k, l = l, k
        return (t,uk,k,ul,l)

    def minimal_d(self, P):
        if len(P) == 0:
            return 0
        d = iter(P).next()[0].total_degree()
        for (t,_,_,_,_) in P:
            if t.total_degree() < d:
                d = t.total_degree()
        return d

    def select(self, P, d):
        return set([p for p in P if p[0].total_degree() == d])

    def compute_spols(self, P):
        poly = self.poly
        sig = self.sig
        spol = self.spol
        is_rewritable = self.is_rewritable
        is_top_reducible = self.is_top_reducible
        add_rule = self.add_rule

        L = self.L

        S = set()
        P = sorted(P, key=lambda x: x[1]*x[2])
        for (t,u,k,v,l) in P:
            if not is_rewritable(u,k) and not is_rewritable(v,l):
                S.add( (u,k) )
                S.add( (v,l) )
        S = sorted(S, key=lambda (u,k): u*sig(k))
        return S

    def reduction(self, S, G):
        """
        INPUT:

        - ``S`` - a list of components of S-polynomials
        - ``G`` - the intermediate Gröbner basis
        """
        L = self.L
        add_rule = self.add_rule
        poly,sig = self.poly, self.sig

        global elim_run_time
        global sym_run_time

        start = time.clock()
        F = self.symbolic_preprocessing(S, G)
        end = time.clock()
        tt = end - start
        #print("\nSymbolic Preprocessing Runtime: " + str(tt))
        sym_run_time += tt

        start = time.clock()
        Ft = self.gauss_elimination(F)
        end = time.clock()
        tt = end - start
        print(" G. Elimination Func. Time: [%6f]")%tt,
        elim_run_time += tt

        Ret = []
        for i, p in enumerate(Ft):
            u,k = F[i]
            if p.lm() == (u*poly(k)).lm():
                continue # ignore unchanged new polynomials
            sigma = u*sig(k)
            L.append( (sigma,p) ) # we have a new polynomial
            add_rule( sigma, len(L)-1 )
            if p != 0:
                Ret.append(len(L) - 1)

        return Ret
        
    def symbolic_preprocessing(self, S, G):
        """
        Add polynomials to the set ``S`` such that all possible
        reductors for all elements in ``S`` are available.

        INPUT:

        - ``S`` - a list of components of S-polynomials
        - ``G`` - the intermediate Gröbner basis indexed in ``L``
        """
        #print("\n\nSym-Processing Started")

        poly = self.poly
        sig = self.sig
        L = self.L
        find_reductor = self.find_reductor

        F = S
        Done = set([(u*poly(k)).lm() for (u,k) in F])

        # the set of all monomials
        M = set([m for (u,k) in F for m in (u*poly(k)).monomials()])

        while M != Done:
            M = sorted(M)
            for i,m in enumerate(M):
                if m not in Done:
                    break
            M = set(M)
            Done.add(m)

            # we need to find the polynomial with the minimal
            # signature which has the monomial m, alternatively we
            # could just use the signature of the polynomial it comes
            # from.
            ms = self.minimal_signature(m, F)
            
            t, g = find_reductor(m, ms, G, F)
            if t!=0:
                F.append( (t,g) )
                M = M.union((t*poly(g)).monomials())
        #print("\n\nSym-Processing Ended")
        return sorted(F, key=lambda (u,k): u*sig(k)) # sort by signature

    def minimal_signature(self, m, F):
        sig, poly = self.sig, self.poly 

        ms = (1, 10**20)
        for (u, k) in F:
            if m in (u*poly(k)).monomials() and u*sig(k) < ms:
                ms = u*sig(k)
        return ms

    def find_reductor(self, m, sig_m, G, F):
        """
        Find a reductor `g_i` for `m` in `G` subject to the F5
        constaints.

        INPUT:
        - ``m`` - a monomial
        - ``sig_m`` - the signature of the smalles f which contains ``m``
        - ``G`` - the intermediate Gröbner basis
        """
        is_rewritable = self.is_rewritable
        is_top_reducible = self.is_top_reducible
        sig = self.sig
        poly = self.poly 

        L = self.L
        R = m.parent()
        for k in chain(*G):
            # Requirement (1) is the normal top-reduction criterion.
            if not R.monomial_divides(poly(k).lm(),m):
                continue
            t =  R.monomial_quotient(m, poly(k).lm())

            # Requirement (2) is making sure that the signature of the
            # reductor is normalized.  Recall that we only want
            # signatures of our polynomials to be normalized - we are
            # discarding non-normalized S-polynomials. If we ignored
            # this condition and our reductor would up having larger
            # signature than S(r_{k_0}), then top_reduction would
            # create a new signed polynomial with our reductor's
            # non-normalized signature. (We might add that, if the
            # reductor had smaller signature than S(r_{k_0}), it would
            # be fine to reduce by it; however, F5 doesn't miss
            # anything by forgoing this opportunity because, by Lemma
            # 3.2.1 (The Normalization Lemma), there will be another
            # normalized reductor with the same head term and smaller
            # signature.
            if is_top_reducible(t * sig(k)[0], G[:sig(k)[1]]):
                continue

            # Requirement (3)
            if is_rewritable(t, k):
                continue

            #  Requirement (4) is a check that makes sure we don't
            #  reduce by something that has the same signature as
            #  m. Recall that we want all signed polynomials used
            #  during the run of F5 to be admissible. If we reduced by
            #  a polynomial that has the same signature, we would be
            #  left with a new polynomial for which we would have no
            #  idea what the signature is. The act of reduction would
            #  have certainly lowered the signature, thus causing
            #  admissibility to be lost. (We will comment on this
            #  requirement later in subsection 3.5. With a little
            #  care, we can loosen this requirement.)
            if sig_m == t*sig(k):
                continue

            # Since no rule is added by S-Polynomials or Symbolic
            # Preprocessing, we need a way to prevent choosing as a
            # reductor the polynomial that needs reduction!
            #if (t,k) in F:
            #    continue

            return t, k
        return 0, -1
        
    def gauss_elimination(self, F1):
        """
        Perform permuted F5-style Gaussian elimination on ``F1``.

        INPUT:

        - ``F1`` - a list of tuples ``(sig, poly, idx)``
        """
        #print('\n\nGuassian Elimination Started')

        global field_size_var
        global print_debug_var

        poly = self.poly

        F = [u*poly(k) for u,k in F1]

        if len(F) == 0:
            return F

        A,v = Sequence(F).coefficient_matrix()
        self.zero_reductions += A.nrows()-A.rank()
        
        if print_debug_var:
            print(type(A[0,0]))
            print('\n\nguass_elimination()\n===============================================================================================')
            print "[%4d x %4d]"%(A.nrows(), A.ncols())
            print(A)
            print('===============================================================================================\n\n')
        
        nrows, ncols = A.nrows(), A.ncols()
        matrix_gpu_list = range(nrows * ncols)
        
        for c in xrange(ncols):
            for r in xrange(nrows):
                temp_var = A[r,c]
                temp_var = float(temp_var)
                matrix_gpu_list[idx2c(r,c,nrows)] = temp_var
        
        if print_debug_var:
            print('\n\nCUDA Before')
            for r in xrange(nrows):
                print ('['),
                for c in xrange(ncols):
                    print "%6f"%(matrix_gpu_list[idx2c(r,c,nrows)]),
                print (']') 
        
        t_start = time.clock()

        if use_cpu_var:
            #Original Algorithm Here
            for c in xrange(ncols):
                for r in xrange(nrows):
                    if A[r,c] != 0:
                        if any(A[r,i] for i in xrange(c)):
                            continue
                        a_inverse = ~A[r,c]
                        #print "A[r,c]: %d, ~A[r,c]: %d\n"%(A[r,c], a_inverse),
                        A.rescale_row(r, a_inverse, c)
                        for i in xrange(r+1,nrows):
                            if A[i,c] != 0:
                                minus_b = -A[i,c]
                                A.add_multiple_of_row(i, r, minus_b, c)
                        break      
        else:        
            #GPU Algorithm Here
            instance = gpuadder.GPUCublas()
            instance.call_cublas_gpu_finite_double(matrix_gpu_list, nrows, ncols, field_size_var)
            R = IntegerModRing(field_size_var)
            
            for c in xrange(ncols):
                for r in xrange(nrows):
                    A[r,c] = IntegerMod(R, matrix_gpu_list[idx2c(r,c,nrows)]) 

        t_end = time.clock()
        
        total_time = t_end - t_start

        print "%4d x %4d, %4d, %4d"%(A.nrows(), A.ncols(), A.rank(), A.nrows()-A.rank()),
        print " Guassian Elimination Time(s): [%6f]"%(total_time),

        F = (A*v).list()

        if print_debug_var:            
            print('\n\nCUDA AFTER [As is]')
            for r in xrange(nrows):
                print ('['),
                for c in xrange(ncols):
                    print "%5s"%(str(
                        int(matrix_gpu_list[idx2c(r,c,nrows)])
                    )
                    ),
                print (']')
                
            print('\n\nOriginal Result\n===============================================================================================')
            print "[%4d x %4d]"%(A.nrows(), A.ncols())
            print(A)
            print('===============================================================================================\n\n')

        return F

    def poly(self, i):
        return self.L[i][1]

    def sig(self, i):
        return self.L[i][0]

    def spol(self, f, g):
        LM = lambda f: f.lm()
        LT = lambda f: f.lt()
        LCM = lambda f,g: f.parent().monomial_lcm(f,g)
        return LCM(LM(f),LM(g)) // LT(f) * f - LCM(LM(f),LM(g)) // LT(g) * g

    def is_top_reducible(self, t, l):
        R = t.parent()
        poly = self.poly
        for g in chain(*l):
            if R.monomial_divides(poly(g).lm(),t):
                return True
        return False

    def add_rule(self, s, k):
        #self.Rules[s[1]].append( (s[0],k) )
        for i,e in enumerate(self.Rules[s[1]]):
            if s[0] < e[0]:
                self.Rules[s[1]] = self.Rules[s[1]][:i] + [(s[0],k)] + self.Rules[s[1]][i:]
                return
        self.Rules[s[1]].append( (s[0],k) )

    def is_rewritable(self, u, k):
        j = self.find_rewriting(u, k)
        return j != k

    def find_rewriting(self, u, k):
        """
        INPUT:
        
        - ``k`` - an index in L
        - ``u`` - a monomial
        """

        divides = lambda x,y: x.parent().monomial_divides(x,y)
        Rules = self.Rules
        mk, v = self.sig(k)
        for ctr in reversed(xrange(len(Rules[v]))):
            mj, j = Rules[v][ctr]
            if divides(mj, u * mk):
                return j
        return k

    def terminate(self, P, G):
        I = Ideal([self.poly(f) for f in chain(*G)]).interreduced_basis()
        for (t,u,k,v,l) in P:
            if not self.is_rewritable(u,k) and not self.is_rewritable(v,l):
                s = self.spol(self.poly(k), self.poly(l))
                if s.reduce(I) != 0:
                    return False
        return True

    def print_stats(self):
        print "                             |L|: %4d"%len(self.L)
        print "                         L is GB: %s"%Ideal([f[1] for f in self.L]).basis_is_groebner()
        print "              reductions to zero: %4d"%self.zero_reductions
        print "                     max. degree: %4d"%self.degree_reached
        print("     symbolic processing runtime: " + str(sym_run_time))
        print("    guassian elimination runtime: " + str(elim_run_time))
        print("                   total runtime: " + str(total_run_time) + "\n")

from UserList import UserList

class Signature(UserList):
    def __init__(self, multiplier, index):
        """
        Create a new signature from the mulitplier and the index.
        """
        UserList.__init__(self, (multiplier, index))
         
    def __lt__(self, other):
        """
        """
        if self[1] < other[1]:
            return True
        elif self[1] > other[1]:
            return False
        else:
            return (self[0] < other[0])

    def __eq__(self, other):
        return self[0] == other[0] and self[1] == other[1]
    
    def __neq__(self, other):
        return self[0] != other[0] or self[1] != other[1]
  
    def __rmul__(self, other):
        if isinstance(self, Signature):
            return Signature(other * self[0], self[1])
        else:
            raise TypeError

    def __hash__(self):
        return hash(self[0]) + hash(self[1])

f5_2  = F5_2()
