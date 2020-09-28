Automated planning
Further information: List of algorithms for automated planning
Combinatorial algorithms
Further information: Combinatorics
General combinatorial algorithms

    Brent's algorithm: finds a cycle in function value iterations using only two iterators
    Floyd's cycle-finding algorithm: finds a cycle in function value iterations
    Gale–Shapley algorithm: solves the stable marriage problem
    Pseudorandom number generators (uniformly distributed—see also List of pseudorandom number generators for other PRNGs with varying degrees of convergence and varying statistical quality):
        ACORN generator
        Blum Blum Shub
        Lagged Fibonacci generator
        Linear congruential generator
        Mersenne Twister

Graph algorithms
Further information: Graph theory and Category:Graph algorithms

    Coloring algorithm: Graph coloring algorithm.
    Hopcroft–Karp algorithm: convert a bipartite graph to a maximum cardinality matching
    Hungarian algorithm: algorithm for finding a perfect matching
    Prüfer coding: conversion between a labeled tree and its Prüfer sequence
    Tarjan's off-line lowest common ancestors algorithm: compute lowest common ancestors for pairs of nodes in a tree
    Topological sort: finds linear order of nodes (e.g. jobs) based on their dependencies.

Graph drawing
Further information: Graph drawing

    Force-based algorithms (also known as force-directed algorithms or spring-based algorithm)
    Spectral layout

Network theory
Further information: Network theory

    Network analysis
        Link analysis
            Girvan–Newman algorithm: detect communities in complex systems
            Web link analysis
                Hyperlink-Induced Topic Search (HITS) (also known as Hubs and authorities)
                PageRank
                TrustRank
    Flow networks
        Dinic's algorithm: is a strongly polynomial algorithm for computing the maximum flow in a flow network.
        Edmonds–Karp algorithm: implementation of Ford–Fulkerson
        Ford–Fulkerson algorithm: computes the maximum flow in a graph
        Karger's algorithm: a Monte Carlo method to compute the minimum cut of a connected graph
        Push–relabel algorithm: computes a maximum flow in a graph

Routing for graphs

    Edmonds' algorithm (also known as Chu–Liu/Edmonds' algorithm): find maximum or minimum branchings
    Euclidean minimum spanning tree: algorithms for computing the minimum spanning tree of a set of points in the plane
    Euclidean shortest path problem: find the shortest path between two points that does not intersect any obstacle
    Longest path problem: find a simple path of maximum length in a given graph
    Minimum spanning tree
        Borůvka's algorithm
        Kruskal's algorithm
        Prim's algorithm
        Reverse-delete algorithm
    Nonblocking minimal spanning switch say, for a telephone exchange
    Shortest path problem
        Bellman–Ford algorithm: computes shortest paths in a weighted graph (where some of the edge weights may be negative)
        Dijkstra's algorithm: computes shortest paths in a graph with non-negative edge weights
        Floyd–Warshall algorithm: solves the all pairs shortest path problem in a weighted, directed graph
        Johnson's algorithm: All pairs shortest path algorithm in sparse weighted directed graph
    Transitive closure problem: find the transitive closure of a given binary relation
    Traveling salesman problem
        Christofides algorithm
        Nearest neighbour algorithm
    Warnsdorff's rule: A heuristic method for solving the Knight's tour problem.

Graph search
Further information: State space search and Graph search algorithm

    A*: special case of best-first search that uses heuristics to improve speed
    B*: a best-first graph search algorithm that finds the least-cost path from a given initial node to any goal node (out of one or more possible goals)
    Backtracking: abandons partial solutions when they are found not to satisfy a complete solution
    Beam search: is a heuristic search algorithm that is an optimization of best-first search that reduces its memory requirement
    Beam stack search: integrates backtracking with beam search
    Best-first search: traverses a graph in the order of likely importance using a priority queue
    Bidirectional search: find the shortest path from an initial vertex to a goal vertex in a directed graph
    Breadth-first search: traverses a graph level by level
    Brute-force search: An exhaustive and reliable search method, but computationally inefficient in many applications.
    D*: an incremental heuristic search algorithm
    Depth-first search: traverses a graph branch by branch
    Dijkstra's algorithm: A special case of A* for which no heuristic function is used
    General Problem Solver: a seminal theorem-proving algorithm intended to work as a universal problem solver machine.
    Iterative deepening depth-first search (IDDFS): a state space search strategy
    Jump point search: An optimization to A* which may reduce computation time by an order of magnitude using further heuristics.
    Lexicographic breadth-first search (also known as Lex-BFS): a linear time algorithm for ordering the vertices of a graph
    Uniform-cost search: a tree search that finds the lowest-cost route where costs vary
    SSS*: state space search traversing a game tree in a best-first fashion similar to that of the A* search algorithm
    F*: Special algorithm to merge the two arrays

Subgraphs

    Cliques
        Bron–Kerbosch algorithm: a technique for finding maximal cliques in an undirected graph
        MaxCliqueDyn maximum clique algorithm: find a maximum clique in an undirected graph
    Strongly connected components
        Path-based strong component algorithm
        Kosaraju's algorithm
        Tarjan's strongly connected components algorithm
    Subgraph isomorphism problem

Sequence algorithms
Further information: Sequences
Approximate sequence matching

    Bitap algorithm: fuzzy algorithm that determines if strings are approximately equal.
    Phonetic algorithms
        Daitch–Mokotoff Soundex: a Soundex refinement which allows matching of Slavic and Germanic surnames
        Double Metaphone: an improvement on Metaphone
        Match rating approach: a phonetic algorithm developed by Western Airlines
        Metaphone: an algorithm for indexing words by their sound, when pronounced in English
        NYSIIS: phonetic algorithm, improves on Soundex
        Soundex: a phonetic algorithm for indexing names by sound, as pronounced in English
    String metrics: compute a similarity or dissimilarity (distance) score between two pairs of text strings
        Damerau–Levenshtein distance compute a distance measure between two strings, improves on Levenshtein distance
        Dice's coefficient (also known as the Dice coefficient): a similarity measure related to the Jaccard index
        Hamming distance: sum number of positions which are different
        Jaro–Winkler distance: is a measure of similarity between two strings
        Levenshtein edit distance: compute a metric for the amount of difference between two sequences
    Trigram search: search for text when the exact syntax or spelling of the target object is not precisely known

Selection algorithms
Main article: Selection algorithm

    Quickselect
    Introselect

Sequence search

    Linear search: finds an item in an unsorted sequence
    Selection algorithm: finds the kth largest item in a sequence
    Ternary search: a technique for finding the minimum or maximum of a function that is either strictly increasing and then strictly decreasing or vice versa
    Sorted lists
        Binary search algorithm: locates an item in a sorted sequence
        Fibonacci search technique: search a sorted sequence using a divide and conquer algorithm that narrows down possible locations with the aid of Fibonacci numbers
        Jump search (or block search): linear search on a smaller subset of the sequence
        Predictive search: binary-like search which factors in magnitude of search term versus the high and low values in the search. Sometimes called dictionary search or interpolated search.
        Uniform binary search: an optimization of the classic binary search algorithm

Sequence merging
Main article: Merge algorithm

    Simple merge algorithm
    k-way merge algorithm
    Union (merge, with elements on the output not repeated)

Sequence permutations
Further information: Permutation

    Fisher–Yates shuffle (also known as the Knuth shuffle): randomly shuffle a finite set
    Schensted algorithm: constructs a pair of Young tableaux from a permutation
    Steinhaus–Johnson–Trotter algorithm (also known as the Johnson–Trotter algorithm): generate permutations by transposing elements
    Heap's permutation generation algorithm: interchange elements to generate next permutation

Sequence alignment

    Dynamic time warping: measure similarity between two sequences which may vary in time or speed
    Hirschberg's algorithm: finds the least cost sequence alignment between two sequences, as measured by their Levenshtein distance
    Needleman–Wunsch algorithm: find global alignment between two sequences
    Smith–Waterman algorithm: find local sequence alignment

Sequence sorting
Main article: Sorting algorithm
Accuracy dispute
	
This article appears to contradict the article Sorting_algorithm#Comparison_of_algorithms. Please see discussion on the linked talk page. (March 2011) (Learn how and when to remove this template message)

    Exchange sorts
        Bubble sort: for each pair of indices, swap the items if out of order
        Cocktail shaker sort or bidirectional bubble sort, a bubble sort traversing the list alternately from front to back and back to front
        Comb sort
        Gnome sort
        Odd–even sort
        Quicksort: divide list into two, with all items on the first list coming before all items on the second list.; then sort the two lists. Often the method of choice
    Humorous or ineffective
        Bogosort
        Stooge sort
    Hybrid
        Flashsort
        Introsort: begin with quicksort and switch to heapsort when the recursion depth exceeds a certain level
        Timsort: adaptative algorithm derived from merge sort and insertion sort. Used in Python 2.3 and up, and Java SE 7.
    Insertion sorts
        Insertion sort: determine where the current item belongs in the list of sorted ones, and insert it there
        Library sort
        Patience sorting
        Shell sort: an attempt to improve insertion sort
        Tree sort (binary tree sort): build binary tree, then traverse it to create sorted list
        Cycle sort: in-place with theoretically optimal number of writes
    Merge sorts
        Merge sort: sort the first and second half of the list separately, then merge the sorted lists
        Slowsort
        Strand sort
    Non-comparison sorts
        Bead sort
        Bucket sort
        Burstsort: build a compact, cache efficient burst trie and then traverse it to create sorted output
        Counting sort
        Pigeonhole sort
        Postman sort: variant of Bucket sort which takes advantage of hierarchical structure
        Radix sort: sorts strings letter by letter
    Selection sorts
        Heapsort: convert the list into a heap, keep removing the largest element from the heap and adding it to the end of the list
        Selection sort: pick the smallest of the remaining elements, add it to the end of the sorted list
        Smoothsort
    Other
        Bitonic sorter
        Pancake sorting
        Spaghetti sort
        Topological sort
    Unknown class
        Samplesort

Subsequences
Further information: Subsequence

    Kadane's algorithm: finds maximum sub-array of any size
    Longest common subsequence problem: Find the longest subsequence common to all sequences in a set of sequences
    Longest increasing subsequence problem: Find the longest increasing subsequence of a given sequence
    Shortest common supersequence problem: Find the shortest supersequence that contains two or more sequences as subsequences

Substrings
Further information: Substring

    Longest common substring problem: find the longest string (or strings) that is a substring (or are substrings) of two or more strings
    Substring search
        Aho–Corasick string matching algorithm: trie based algorithm for finding all substring matches to any of a finite set of strings
        Boyer–Moore string-search algorithm: amortized linear (sublinear in most times) algorithm for substring search
        Boyer–Moore–Horspool algorithm: Simplification of Boyer–Moore
        Knuth–Morris–Pratt algorithm: substring search which bypasses reexamination of matched characters
        Rabin–Karp string search algorithm: searches multiple patterns efficiently
        Zhu–Takaoka string matching algorithm: a variant of Boyer–Moore
    Ukkonen's algorithm: a linear-time, online algorithm for constructing suffix trees
    Matching wildcards
        Rich Salz' wildmat: a widely used open-source recursive algorithm
        Krauss matching wildcards algorithm: an open-source non-recursive algorithm

Computational mathematics
Further information: Computational mathematics
See also: Combinatorial algorithms and Computational science
Abstract algebra
Further information: Abstract algebra

    Chien search: a recursive algorithm for determining roots of polynomials defined over a finite field
    Schreier–Sims algorithm: computing a base and strong generating set (BSGS) of a permutation group
    Todd–Coxeter algorithm: Procedure for generating cosets.

Computer algebra
Further information: Computer algebra

    Buchberger's algorithm: finds a Gröbner basis
    Cantor–Zassenhaus algorithm: factor polynomials over finite fields
    Faugère F4 algorithm: finds a Gröbner basis (also mentions the F5 algorithm)
    Gosper's algorithm: find sums of hypergeometric terms that are themselves hypergeometric terms
    Knuth–Bendix completion algorithm: for rewriting rule systems
    Multivariate division algorithm: for polynomials in several indeterminates
    Pollard's kangaroo algorithm (also known as Pollard's lambda algorithm ): an algorithm for solving the discrete logarithm problem
    Polynomial long division: an algorithm for dividing a polynomial by another polynomial of the same or lower degree
    Risch algorithm: an algorithm for the calculus operation of indefinite integration (i.e. finding antiderivatives)

Geometry
Main category: Geometric algorithms
Further information: Computational geometry

    Closest pair problem: find the pair of points (from a set of points) with the smallest distance between them
    Collision detection algorithms: check for the collision or intersection of two given solids
    Cone algorithm: identify surface points
    Convex hull algorithms: determining the convex hull of a set of points
        Graham scan
        Quickhull
        Gift wrapping algorithm or Jarvis march
        Chan's algorithm
        Kirkpatrick–Seidel algorithm
    Euclidean distance transform: computes the distance between every point in a grid and a discrete collection of points.
    Geometric hashing: a method for efficiently finding two-dimensional objects represented by discrete points that have undergone an affine transformation
    Gilbert–Johnson–Keerthi distance algorithm: determining the smallest distance between two convex shapes.
    Jump-and-Walk algorithm: an algorithm for point location in triangulations
    Laplacian smoothing: an algorithm to smooth a polygonal mesh
    Line segment intersection: finding whether lines intersect, usually with a sweep line algorithm
        Bentley–Ottmann algorithm
        Shamos–Hoey algorithm
    Minimum bounding box algorithms: find the oriented minimum bounding box enclosing a set of points
    Nearest neighbor search: find the nearest point or points to a query point
    Point in polygon algorithms: tests whether a given point lies within a given polygon
    Point set registration algorithms: finds the transformation between two point sets to optimally align them.
    Rotating calipers: determine all antipodal pairs of points and vertices on a convex polygon or convex hull.
    Shoelace algorithm: determine the area of a polygon whose vertices are described by ordered pairs in the plane
    Triangulation
        Delaunay triangulation
            Ruppert's algorithm (also known as Delaunay refinement): create quality Delaunay triangulations
            Chew's second algorithm: create quality constrained Delaunay triangulations
        Marching triangles: reconstruct two-dimensional surface geometry from an unstructured point cloud
        Polygon triangulation algorithms: decompose a polygon into a set of triangles
        Voronoi diagrams, geometric dual of Delaunay triangulation
            Bowyer–Watson algorithm: create voronoi diagram in any number of dimensions
            Fortune's Algorithm: create voronoi diagram
        Quasitriangulation

Number theoretic algorithms
Further information: Number theory

    Binary GCD algorithm: Efficient way of calculating GCD.
    Booth's multiplication algorithm
    Chakravala method: a cyclic algorithm to solve indeterminate quadratic equations, including Pell's equation
    Discrete logarithm:
        Baby-step giant-step
        Index calculus algorithm
        Pollard's rho algorithm for logarithms
        Pohlig–Hellman algorithm
    Euclidean algorithm: computes the greatest common divisor
    Extended Euclidean algorithm: Also solves the equation ax + by = c.
    Integer factorization: breaking an integer into its prime factors
        Congruence of squares
        Dixon's algorithm
        Fermat's factorization method
        General number field sieve
        Lenstra elliptic curve factorization
        Pollard's p − 1 algorithm
        Pollard's rho algorithm
        prime factorization algorithm
        Quadratic sieve
        Shor's algorithm
        Special number field sieve
        Trial division
    Multiplication algorithms: fast multiplication of two numbers
        Karatsuba algorithm
        Schönhage–Strassen algorithm
        Toom–Cook multiplication
    Modular square root: computing square roots modulo a prime number
        Tonelli–Shanks algorithm
        Cipolla's algorithm
        Berlekamp's root finding algorithm
    Odlyzko–Schönhage algorithm: calculates nontrivial zeroes of the Riemann zeta function
    Lenstra–Lenstra–Lovász algorithm (also known as LLL algorithm): find a short, nearly orthogonal lattice basis in polynomial time
    Primality tests: determining whether a given number is prime
        AKS primality test
        Baillie-PSW primality test
        Fermat primality test
        Lucas primality test
        Miller–Rabin primality test
        Sieve of Atkin
        Sieve of Eratosthenes
        Sieve of Sundaram

Numerical algorithms
Further information: Numerical analysis and List of numerical analysis topics
Differential equation solving
Further information: Differential equation

    Euler method
    Backward Euler method
    Trapezoidal rule (differential equations)
    Linear multistep methods
    Runge–Kutta methods
        Euler integration
    Multigrid methods (MG methods), a group of algorithms for solving differential equations using a hierarchy of discretizations
    Partial differential equation:
        Finite difference method
        Crank–Nicolson method for diffusion equations
        Lax-Wendroff for wave equations
    Verlet integration (French pronunciation: ​[vɛʁˈlɛ]): integrate Newton's equations of motion

Elementary and special functions
Further information: Special functions

    Computation of π:
        Borwein's algorithm: an algorithm to calculate the value of 1/π
        Gauss–Legendre algorithm: computes the digits of pi
        Chudnovsky algorithm: A fast method for calculating the digits of π
        Bailey–Borwein–Plouffe formula: (BBP formula) a spigot algorithm for the computation of the nth binary digit of π
    Division algorithms: for computing quotient and/or remainder of two numbers
        Long division
        Restoring division
        Non-restoring division
        SRT division
        Newton–Raphson division: uses Newton's method to find the reciprocal of D, and multiply that reciprocal by N to find the final quotient Q.
        Goldschmidt division
    Hyperbolic and Trigonometric Functions:
        BKM algorithm: compute elementary functions using a table of logarithms
        CORDIC: compute hyperbolic and trigonometric functions using a table of arctangents
    Exponentiation:
        Addition-chain exponentiation: exponentiation by positive integer powers that requires a minimal number of multiplications
        Exponentiating by squaring: an algorithm used for the fast computation of large integer powers of a number
    Montgomery reduction: an algorithm that allows modular arithmetic to be performed efficiently when the modulus is large
    Multiplication algorithms: fast multiplication of two numbers
        Booth's multiplication algorithm: a multiplication algorithm that multiplies two signed binary numbers in two's complement notation
        Fürer's algorithm: an integer multiplication algorithm for very large numbers possessing a very low asymptotic complexity
        Karatsuba algorithm: an efficient procedure for multiplying large numbers
        Schönhage–Strassen algorithm: an asymptotically fast multiplication algorithm for large integers
        Toom–Cook multiplication: (Toom3) a multiplication algorithm for large integers
    Multiplicative inverse Algorithms: for computing a number's multiplicative inverse (reciprocal).
        Newton's method
    Rounding functions: the classic ways to round numbers
    Spigot algorithm: A way to compute the value of a mathematical constant without knowing preceding digits
    Square and Nth root of a number:
        Alpha max plus beta min algorithm: an approximation of the square-root of the sum of two squares
        Methods of computing square roots
        nth root algorithm
        Shifting nth-root algorithm: digit by digit root extraction
    Summation:
        Binary splitting: a divide and conquer technique which speeds up the numerical evaluation of many types of series with rational terms
        Kahan summation algorithm: a more accurate method of summing floating-point numbers
    Unrestricted algorithm

Geometric

    Filtered back-projection: efficiently compute the inverse 2-dimensional Radon transform.
    Level set method (LSM): a numerical technique for tracking interfaces and shapes

Interpolation and extrapolation
Further information: Interpolation and Extrapolation

    Birkhoff interpolation: an extension of polynomial interpolation
    Cubic interpolation
    Hermite interpolation
    Lagrange interpolation: interpolation using Lagrange polynomials
    Linear interpolation: a method of curve fitting using linear polynomials
    Monotone cubic interpolation: a variant of cubic interpolation that preserves monotonicity of the data set being interpolated.
    Multivariate interpolation
        Bicubic interpolation, a generalization of cubic interpolation to two dimensions
        Bilinear interpolation: an extension of linear interpolation for interpolating functions of two variables on a regular grid
        Lanczos resampling ("Lanzosh"): a multivariate interpolation method used to compute new values for any digitally sampled data
        Nearest-neighbor interpolation
        Tricubic interpolation, a generalization of cubic interpolation to three dimensions
    Pareto interpolation: a method of estimating the median and other properties of a population that follows a Pareto distribution.
    Polynomial interpolation
        Neville's algorithm
    Spline interpolation: Reduces error with Runge's phenomenon.
        De Boor algorithm: B-splines
        De Casteljau's algorithm: Bézier curves
    Trigonometric interpolation

Linear algebra
Further information: Numerical linear algebra

    Eigenvalue algorithms
        Arnoldi iteration
        Inverse iteration
        Jacobi method
        Lanczos iteration
        Power iteration
        QR algorithm
        Rayleigh quotient iteration
    Gram–Schmidt process: orthogonalizes a set of vectors
    Matrix multiplication algorithms
        Cannon's algorithm: a distributed algorithm for matrix multiplication especially suitable for computers laid out in an N × N mesh
        Coppersmith–Winograd algorithm: square matrix multiplication
        Freivalds' algorithm: a randomized algorithm used to verify matrix multiplication
        Strassen algorithm: faster matrix multiplication

    Solving systems of linear equations
        Biconjugate gradient method: solves systems of linear equations
        Conjugate gradient: an algorithm for the numerical solution of particular systems of linear equations
        Gaussian elimination
        Gauss–Jordan elimination: solves systems of linear equations
        Gauss–Seidel method: solves systems of linear equations iteratively
        Levinson recursion: solves equation involving a Toeplitz matrix
        Stone's method: also known as the strongly implicit procedure or SIP, is an algorithm for solving a sparse linear system of equations
        Successive over-relaxation (SOR): method used to speed up convergence of the Gauss–Seidel method
        Tridiagonal matrix algorithm (Thomas algorithm): solves systems of tridiagonal equations
    Sparse matrix algorithms
        Cuthill–McKee algorithm: reduce the bandwidth of a symmetric sparse matrix
        Minimum degree algorithm: permute the rows and columns of a symmetric sparse matrix before applying the Cholesky decomposition
        Symbolic Cholesky decomposition: Efficient way of storing sparse matrix

Monte Carlo
Further information: Monte Carlo method

    Gibbs sampling: generate a sequence of samples from the joint probability distribution of two or more random variables
    Hybrid Monte Carlo: generate a sequence of samples using Hamiltonian weighted Markov chain Monte Carlo, from a probability distribution which is difficult to sample directly.
    Metropolis–Hastings algorithm: used to generate a sequence of samples from the probability distribution of one or more variables
    Wang and Landau algorithm: an extension of Metropolis–Hastings algorithm sampling

Numerical integration
Further information: Numerical integration

    MISER algorithm: Monte Carlo simulation, numerical integration

Root finding
Main article: Root-finding algorithm

    Bisection method
    False position method: approximates roots of a function
    Newton's method: finds zeros of functions with calculus
    Halley's method: uses first and second derivatives
    Secant method: 2-point, 1-sided
    False position method and Illinois method: 2-point, bracketing
    Ridder's method: 3-point, exponential scaling
    Muller's method: 3-point, quadratic interpolation

Optimization algorithms
Main article: Mathematical optimization

    Alpha–beta pruning: search to reduce number of nodes in minimax algorithm
    Branch and bound
    Bruss algorithm: see odds algorithm
    Chain matrix multiplication
    Combinatorial optimization: optimization problems where the set of feasible solutions is discrete
        Greedy randomized adaptive search procedure (GRASP): successive constructions of a greedy randomized solution and subsequent iterative improvements of it through a local search
        Hungarian method: a combinatorial optimization algorithm which solves the assignment problem in polynomial time
    Constraint satisfaction
        General algorithms for the constraint satisfaction
            AC-3 algorithm
            Difference map algorithm
            Min conflicts algorithm
        Chaff algorithm: an algorithm for solving instances of the boolean satisfiability problem
        Davis–Putnam algorithm: check the validity of a first-order logic formula
        Davis–Putnam–Logemann–Loveland algorithm (DPLL): an algorithm for deciding the satisfiability of propositional logic formula in conjunctive normal form, i.e. for solving the CNF-SAT problem
        Exact cover problem
            Algorithm X: a nondeterministic algorithm
            Dancing Links: an efficient implementation of Algorithm X
    Cross-entropy method: a general Monte Carlo approach to combinatorial and continuous multi-extremal optimization and importance sampling
    Differential evolution
    Dynamic Programming: problems exhibiting the properties of overlapping subproblems and optimal substructure
    Ellipsoid method: is an algorithm for solving convex optimization problems
    Evolutionary computation: optimization inspired by biological mechanisms of evolution
        Evolution strategy
        Gene expression programming
        Genetic algorithms
            Fitness proportionate selection - also known as roulette-wheel selection
            Stochastic universal sampling
            Truncation selection
            Tournament selection
        Memetic algorithm
        Swarm intelligence
            Ant colony optimization
            Bees algorithm: a search algorithm which mimics the food foraging behavior of swarms of honey bees
            Particle swarm
    golden section search: an algorithm for finding the maximum of a real function
    Gradient descent
    Harmony search (HS): a metaheuristic algorithm mimicking the improvisation process of musicians
    Interior point method
    Linear programming
        Benson's algorithm: an algorithm for solving linear vector optimization problems
        Dantzig–Wolfe decomposition: an algorithm for solving linear programming problems with special structure
        Delayed column generation
        Integer linear programming: solve linear programming problems where some or all the unknowns are restricted to integer values
            Branch and cut
            Cutting-plane method
        Karmarkar's algorithm: The first reasonably efficient algorithm that solves the linear programming problem in polynomial time.
        Simplex algorithm: An algorithm for solving linear programming problems
    Line search
    Local search: a metaheuristic for solving computationally hard optimization problems
        Random-restart hill climbing
        Tabu search
    Minimax used in game programming
    Nearest neighbor search (NNS): find closest points in a metric space
        Best Bin First: find an approximate solution to the Nearest neighbor search problem in very-high-dimensional spaces
    Newton's method in optimization
    Nonlinear optimization
        BFGS method: A nonlinear optimization algorithm
        Gauss–Newton algorithm: An algorithm for solving nonlinear least squares problems.
        Levenberg–Marquardt algorithm: An algorithm for solving nonlinear least squares problems.
        Nelder–Mead method (downhill simplex method): A nonlinear optimization algorithm
    Odds algorithm (Bruss algorithm) : Finds the optimal strategy to predict a last specific event in a random sequence event
    Simulated annealing
    Stochastic tunneling
    Subset sum algorithm

Computational science
Further information: Computational science
Astronomy
Main article: Astronomical algorithms

    Doomsday algorithm: day of the week
    Zeller's congruence is an algorithm to calculate the day of the week for any Julian or Gregorian calendar date
    various Easter algorithms are used to calculate the day of Easter

Bioinformatics
Further information: Bioinformatics
See also: Sequence alignment algorithms

    Basic Local Alignment Search Tool also known as BLAST: an algorithm for comparing primary biological sequence information
    Kabsch algorithm: calculate the optimal alignment of two sets of points in order to compute the root mean squared deviation between two protein structures.
    Velvet: a set of algorithms manipulating de Bruijn graphs for genomic sequence assembly
    Sorting by signed reversals: an algorithm for understanding genomic evolution.
    Maximum parsimony (phylogenetics): an algorithm for finding the simplest phylogenetic tree to explain a given character matrix.
    UPGMA: a distance-based phylogenetic tree construction algorithm.

Geoscience
Further information: Geoscience

    Vincenty's formulae: a fast algorithm to calculate the distance between two latitude/longitude points on an ellipsoid
    Geohash: a public domain algorithm that encodes a decimal latitude/longitude pair as a hash string

Linguistics
Further information: Computational linguistics and Natural language processing

    Lesk algorithm: word sense disambiguation
    Stemming algorithm: a method of reducing words to their stem, base, or root form
    Sukhotin's algorithm: a statistical classification algorithm for classifying characters in a text as vowels or consonants

Medicine
Further information: Medical algorithms

    ESC algorithm for the diagnosis of heart failure
    Manning Criteria for irritable bowel syndrome
    Pulmonary embolism diagnostic algorithms
    Texas Medication Algorithm Project

Physics
Further information: Computational physics

    Constraint algorithm: a class of algorithms for satisfying constraints for bodies that obey Newton's equations of motion
    Demon algorithm: a Monte Carlo method for efficiently sampling members of a microcanonical ensemble with a given energy
    Featherstone's algorithm: compute the effects of forces applied to a structure of joints and links
    Ground state approximation
        Variational method
            Ritz method
    n-body problems
        Barnes–Hut simulation: Solves the n-body problem in an approximate way that has the order O(n log n) instead of O(n2) as in a direct-sum simulation.
        Fast multipole method (FMM): speeds up the calculation of long-ranged forces
    Rainflow-counting algorithm: Reduces a complex stress history to a count of elementary stress-reversals for use in fatigue analysis
    Sweep and prune: a broad phase algorithm used during collision detection to limit the number of pairs of solids that need to be checked for collision
    VEGAS algorithm: a method for reducing error in Monte Carlo simulations

Statistics
Further information: Computational statistics

    Algorithms for calculating variance: avoiding instability and numerical overflow
    Approximate counting algorithm: Allows counting large number of events in a small register
    Bayesian statistics
        Nested sampling algorithm: a computational approach to the problem of comparing models in Bayesian statistics
    Clustering Algorithms
        Average-linkage clustering: a simple agglomerative clustering algorithm
        Canopy clustering algorithm: an unsupervised pre-clustering algorithm related to the K-means algorithm
        Complete-linkage clustering: a simple agglomerative clustering algorithm
        DBSCAN: a density based clustering algorithm
        Expectation-maximization algorithm
        Fuzzy clustering: a class of clustering algorithms where each point has a degree of belonging to clusters
            Fuzzy c-means
            FLAME clustering (Fuzzy clustering by Local Approximation of MEmberships): define clusters in the dense parts of a dataset and perform cluster assignment solely based on the neighborhood relationships among objects
        KHOPCA clustering algorithm: a local clustering algorithm, which produces hierarchical multi-hop clusters in static and mobile environments.
        k-means clustering: cluster objects based on attributes into partitions
        k-means++: a variation of this, using modified random seeds
        k-medoids: similar to k-means, but chooses datapoints or medoids as centers
        Linde–Buzo–Gray algorithm: a vector quantization algorithm to derive a good codebook
        Lloyd's algorithm (Voronoi iteration or relaxation): group data points into a given number of categories, a popular algorithm for k-means clustering
        OPTICS: a density based clustering algorithm with a visual evaluation method
        Single-linkage clustering: a simple agglomerative clustering algorithm
        SUBCLU: a subspace clustering algorithm
        Ward's method : an agglomerative clustering algorithm, extended to more general Lance–Williams algorithms
        WACA clustering algorithm: a local clustering algorithm with potentially multi-hop structures; for dynamic networks
    Estimation Theory
        Expectation-maximization algorithm A class of related algorithms for finding maximum likelihood estimates of parameters in probabilistic models
            Ordered subset expectation maximization (OSEM): used in medical imaging for positron emission tomography, single photon emission computed tomography and X-ray computed tomography.
        Odds algorithm (Bruss algorithm) Optimal online search for distinguished value in sequential random input
        Kalman filter: estimate the state of a linear dynamic system from a series of noisy measurements
    False nearest neighbor algorithm (FNN) estimates fractal dimension
    Hidden Markov model
        Baum–Welch algorithm: compute maximum likelihood estimates and posterior mode estimates for the parameters of a hidden Markov model
        Forward-backward algorithm a dynamic programming algorithm for computing the probability of a particular observation sequence
        Viterbi algorithm: find the most likely sequence of hidden states in a hidden Markov model
    Partial least squares regression: finds a linear model describing some predicted variables in terms of other observable variables
    Queuing theory
        Buzen's algorithm: an algorithm for calculating the normalization constant G(K) in the Gordon–Newell theorem
    RANSAC (an abbreviation for "RANdom SAmple Consensus"): an iterative method to estimate parameters of a mathematical model from a set of observed data which contains outliers
    Scoring algorithm: is a form of Newton's method used to solve maximum likelihood equations numerically
    Yamartino method: calculate an approximation to the standard deviation σθ of wind direction θ during a single pass through the incoming data
    Ziggurat algorithm: generate random numbers from a non-uniform distribution

Computer science
Further information: Computer science
Computer architecture
Further information: Computer architecture

    Tomasulo algorithm: allows sequential instructions that would normally be stalled due to certain dependencies to execute non-sequentially

Computer graphics
Further information: Computer graphics

    Clipping
        Line clipping
            Cohen–Sutherland
            Cyrus–Beck
            Fast-clipping
            Liang–Barsky
            Nicholl–Lee–Nicholl
        Polygon clipping
            Sutherland–Hodgman
            Vatti
            Weiler–Atherton
    Contour lines and Isosurfaces
        Marching cubes: extract a polygonal mesh of an isosurface from a three-dimensional scalar field (sometimes called voxels)
        Marching squares: generate contour lines for a two-dimensional scalar field
        Marching tetrahedrons: an alternative to Marching cubes
    Discrete Green's Theorem: is an algorithm for computing double integral over a generalized rectangular domain in constant time. It is a natural extension to the summed area table algorithm
    Flood fill: fills a connected region of a multi-dimensional array with a specified symbol
    Global illumination algorithms: Considers direct illumination and reflection from other objects.
        Ambient occlusion
        Beam tracing
        Cone tracing
        Image-based lighting
        Metropolis light transport
        Path tracing
        Photon mapping
        Radiosity
        Ray tracing
    Hidden surface removal or Visual surface determination
        Newell's algorithm: eliminate polygon cycles in the depth sorting required in hidden surface removal
        Painter's algorithm: detects visible parts of a 3-dimensional scenery
        Scanline rendering: constructs an image by moving an imaginary line over the image
        Warnock algorithm
    Line Drawing: graphical algorithm for approximating a line segment on discrete graphical media.
        Bresenham's line algorithm: plots points of a 2-dimensional array to form a straight line between 2 specified points (uses decision variables)
        DDA line algorithm: plots points of a 2-dimensional array to form a straight line between 2 specified points (uses floating-point math)
        Xiaolin Wu's line algorithm: algorithm for line antialiasing.
    Midpoint circle algorithm: an algorithm used to determine the points needed for drawing a circle
    Ramer–Douglas–Peucker algorithm: Given a 'curve' composed of line segments to find a curve not too dissimilar but that has fewer points
    Shading
        Gouraud shading: an algorithm to simulate the differing effects of light and colour across the surface of an object in 3D computer graphics
        Phong shading: an algorithm to interpolate surface normal-vectors for surface shading in 3D computer graphics
    Slerp (spherical linear interpolation): quaternion interpolation for the purpose of animating 3D rotation
    Summed area table (also known as an integral image): an algorithm for computing the sum of values in a rectangular subset of a grid in constant time

Cryptography
Further information: Cryptography and Topics in cryptography

    Asymmetric (public key) encryption:
        ElGamal
        Elliptic curve cryptography
        MAE1
        NTRUEncrypt
        RSA
    Digital signatures (asymmetric authentication):
        DSA, and its variants:
            ECDSA and Deterministic ECDSA
            EdDSA (Ed25519)
        RSA
    Cryptographic hash functions (see also the section on message authentication codes):
        BLAKE
        MD5 – Note that there is now a method of generating collisions for MD5
        RIPEMD-160
        SHA-1 – Note that there is now a method of generating collisions for SHA-1
        SHA-2 (SHA-224, SHA-256, SHA-384, SHA-512)
        SHA-3 (SHA3-224, SHA3-256, SHA3-384, SHA3-512, SHAKE128, SHAKE256)
        Tiger (TTH), usually used in Tiger tree hashes
        WHIRLPOOL
    Cryptographically secure pseudo-random number generators
        Blum Blum Shub - based on the hardness of factorization
        Fortuna, intended as an improvement on Yarrow algorithm
        Linear-feedback shift register (note: many LFSR-based algorithms are weak or have been broken)
        Yarrow algorithm
    Key exchange
        Diffie–Hellman key exchange
        Elliptic-curve Diffie-Hellman (ECDH)
    Key derivation functions, often used for password hashing and key stretching
        bcrypt
        PBKDF2
        scrypt
        Argon2
    Message authentication codes (symmetric authentication algorithms, which take a key as a parameter):
        HMAC: keyed-hash message authentication
        Poly1305
        SipHash
    Secret sharing, Secret Splitting, Key Splitting, M of N algorithms
        Blakey's Scheme
        Shamir's Scheme
    Symmetric (secret key) encryption:
        Advanced Encryption Standard (AES), winner of NIST competition, also known as Rijndael
        Blowfish
        Twofish
        Threefish
        Data Encryption Standard (DES), sometimes DE Algorithm, winner of NBS selection competition, replaced by AES for most purposes
        IDEA
        RC4 (cipher)
        Tiny Encryption Algorithm (TEA)
        Salsa20, and its updated variant ChaCha20
    Post-quantum cryptography
    Proof-of-work algorithms

Digital logic

    Boolean minimization
        Quine–McCluskey algorithm: Also called as Q-M algorithm, programmable method for simplifying the boolean equations.
        Petrick's method: Another algorithm for boolean simplification.
        Espresso heuristic logic minimizer: Fast algorithm for boolean function minimization.

Machine learning and statistical classification
Main article: List of machine learning algorithms
Further information: Machine learning and Statistical classification

    ALOPEX: a correlation-based machine-learning algorithm
    Association rule learning: discover interesting relations between variables, used in data mining
        Apriori algorithm
        Eclat algorithm
        FP-growth algorithm
        One-attribute rule
        Zero-attribute rule
    Boosting (meta-algorithm): Use many weak learners to boost effectiveness
        AdaBoost: adaptive boosting
        BrownBoost:a boosting algorithm that may be robust to noisy datasets
        LogitBoost: logistic regression boosting
        LPBoost: linear programming boosting
    Bootstrap aggregating (bagging): technique to improve stability and classification accuracy
    Computer Vision
        Grabcut based on Graph cuts
    Decision Trees
        C4.5 algorithm: an extension to ID3
        ID3 algorithm (Iterative Dichotomiser 3): Use heuristic to generate small decision trees
    Clustering: Class of unsupervised learning algorithms for grouping and bucketing related input vector.
        k-nearest neighbors (k-NN): a method for classifying objects based on closest training examples in the feature space
    Linde–Buzo–Gray algorithm: a vector quantization algorithm used to derive a good codebook
    Locality-sensitive hashing (LSH): a method of performing probabilistic dimension reduction of high-dimensional data
    Neural Network
        Backpropagation: A supervised learning method which requires a teacher that knows, or can calculate, the desired output for any given input
        Hopfield net: a Recurrent neural network in which all connections are symmetric
        Perceptron: the simplest kind of feedforward neural network: a linear classifier.
        Pulse-coupled neural networks (PCNN): Neural models proposed by modeling a cat's visual cortex and developed for high-performance biomimetic image processing.
        Radial basis function network: an artificial neural network that uses radial basis functions as activation functions
        Self-organizing map: an unsupervised network that produces a low-dimensional representation of the input space of the training samples
    Random forest: classify using many decision trees
    Reinforcement Learning:
        Q-learning: learn an action-value function that gives the expected utility of taking a given action in a given state and following a fixed policy thereafter
        State-Action-Reward-State-Action (SARSA): learn a Markov decision process policy
        Temporal difference learning
    Relevance Vector Machine (RVM): similar to SVM, but provides probabilistic classification
    Supervised Learning: Learning by examples (labelled data-set split into training-set and test-set)
    Support Vector Machines (SVM): a set of methods which divide multidimensional data by finding a dividing hyperplane with the maximum margin between the two sets
        Structured SVM: allows training of a classifier for general structured output labels.
    Winnow algorithm: related to the perceptron, but uses a multiplicative weight-update scheme

Programming language theory
Further information: Programming language theory

    C3 linearization: an algorithm used primarily to obtain a consistent linearization of a multiple inheritance hierarchy in object-oriented programming
    Chaitin's algorithm: a bottom-up, graph coloring register allocation algorithm that uses cost/degree as its spill metric
    Hindley–Milner type inference algorithm
    Rete algorithm: an efficient pattern matching algorithm for implementing production rule systems
    Sethi-Ullman algorithm: generate optimal code for arithmetic expressions

Parsing
Further information: Parsing

    CYK algorithm: An O(n3) algorithm for parsing context-free grammars in Chomsky normal form
    Earley parser: Another O(n3) algorithm for parsing any context-free grammar
    GLR parser:An algorithm for parsing any context-free grammar by Masaru Tomita. It is tuned for deterministic grammars, on which it performs almost linear time and O(n3) in worst case.
    Inside-outside algorithm: An O(n3) algorithm for re-estimating production probabilities in probabilistic context-free grammars
    LL parser: A relatively simple linear time parsing algorithm for a limited class of context-free grammars
    LR parser: A more complex linear time parsing algorithm for a larger class of context-free grammars. Variants:
        Canonical LR parser
        LALR (Look-ahead LR) parser
        Operator-precedence parser
        SLR (Simple LR) parser
        Simple precedence parser
    Packrat parser: A linear time parsing algorithm supporting some context-free grammars and parsing expression grammars
    Recursive descent parser: A top-down parser suitable for LL(k) grammars
    Shunting yard algorithm: convert an infix-notation math expression to postfix
    Pratt parser
    Lexical analysis

Quantum algorithms
Further information: Quantum algorithm

    Deutsch–Jozsa algorithm: criterion of balance for Boolean function
    Grover's algorithm: provides quadratic speedup for many search problems
    Shor's algorithm: provides exponential speedup (relative to currently known non-quantum algorithms) for factoring a number
    Simon's algorithm: provides a provably exponential speedup (relative to any non-quantum algorithm) for a black-box problem

Theory of computation and automata
Further information: Theory of computation

    Hopcroft's algorithm, Moore's algorithm, and Brzozowski's algorithm: algorithms for minimizing the number of states in a deterministic finite automaton
    Powerset construction: Algorithm to convert nondeterministic automaton to deterministic automaton.
    Tarski–Kuratowski algorithm: a non-deterministic algorithm which provides an upper bound for the complexity of formulas in the arithmetical hierarchy and analytical hierarchy

Information theory and signal processing
Main articles: Information theory and Signal processing
Coding theory
Further information: Coding theory
Error detection and correction
Further information: Error detection and correction

    BCH Codes
        Berlekamp–Massey algorithm
        Peterson–Gorenstein–Zierler algorithm
        Reed–Solomon error correction
    BCJR algorithm: decoding of error correcting codes defined on trellises (principally convolutional codes)
    Forward error correction
    Gray code
    Hamming codes
        Hamming(7,4): a Hamming code that encodes 4 bits of data into 7 bits by adding 3 parity bits
        Hamming distance: sum number of positions which are different
        Hamming weight (population count): find the number of 1 bits in a binary word
    Redundancy checks
        Adler-32
        Cyclic redundancy check
        Damm algorithm
        Fletcher's checksum
        Longitudinal redundancy check (LRC)
        Luhn algorithm: a method of validating identification numbers
        Luhn mod N algorithm: extension of Luhn to non-numeric characters
        Parity: simple/fast error detection technique
        Verhoeff algorithm

Lossless compression algorithms
Main page: Lossless compression algorithms

    Burrows–Wheeler transform: preprocessing useful for improving lossless compression
    Context tree weighting
    Delta encoding: aid to compression of data in which sequential data occurs frequently
    Dynamic Markov compression: Compression using predictive arithmetic coding
    Dictionary coders
        Byte pair encoding (BPE)
        DEFLATE
        Lempel–Ziv
            LZ77 and LZ78
            Lempel–Ziv Jeff Bonwick (LZJB)
            Lempel–Ziv–Markov chain algorithm (LZMA)
            Lempel–Ziv–Oberhumer (LZO): speed oriented
            Lempel–Ziv–Stac (LZS)
            Lempel–Ziv–Storer–Szymanski (LZSS)
            Lempel–Ziv–Welch (LZW)
            LZWL: syllable-based variant
            LZX
            Lempel–Ziv Ross Williams (LZRW)
    Entropy encoding: coding scheme that assigns codes to symbols so as to match code lengths with the probabilities of the symbols
        Arithmetic coding: advanced entropy coding
            Range encoding: same as arithmetic coding, but looked at in a slightly different way
        Huffman coding: simple lossless compression taking advantage of relative character frequencies
            Adaptive Huffman coding: adaptive coding technique based on Huffman coding
            Package-merge algorithm: Optimizes Huffman coding subject to a length restriction on code strings
        Shannon–Fano coding
        Shannon–Fano–Elias coding: precursor to arithmetic encoding[1]
    Entropy coding with known entropy characteristics
        Golomb coding: form of entropy coding that is optimal for alphabets following geometric distributions
        Rice coding: form of entropy coding that is optimal for alphabets following geometric distributions
        Truncated binary encoding
        Unary coding: code that represents a number n with n ones followed by a zero
        Universal codes: encodes positive integers into binary code words
            Elias delta, gamma, and omega coding
            Exponential-Golomb coding
            Fibonacci coding
            Levenshtein coding
    Fast Efficient & Lossless Image Compression System (FELICS): a lossless image compression algorithm
    Incremental encoding: delta encoding applied to sequences of strings
    Prediction by partial matching (PPM): an adaptive statistical data compression technique based on context modeling and prediction
    Run-length encoding: lossless data compression taking advantage of strings of repeated characters
    SEQUITUR algorithm: lossless compression by incremental grammar inference on a string

Lossy compression algorithms
Main page: Lossy compression algorithms

    3Dc: a lossy data compression algorithm for normal maps
    Audio and Speech compression
        A-law algorithm: standard companding algorithm
        Code-excited linear prediction (CELP): low bit-rate speech compression
        Linear predictive coding (LPC): lossy compression by representing the spectral envelope of a digital signal of speech in compressed form
        Mu-law algorithm: standard analog signal compression or companding algorithm
        Warped Linear Predictive Coding (WLPC)
    Image compression
        Block Truncation Coding (BTC): a type of lossy image compression technique for greyscale images
        Embedded Zerotree Wavelet (EZW)
        Fast Cosine Transform algorithms (FCT algorithms): compute Discrete Cosine Transform (DCT) efficiently
        Fractal compression: method used to compress images using fractals
        Set Partitioning in Hierarchical Trees (SPIHT)
        Wavelet compression: form of data compression well suited for image compression (sometimes also video compression and audio compression)
    Transform coding: type of data compression for "natural" data like audio signals or photographic images
    Video compression
    Vector quantization: technique often used in lossy data compression

Digital signal processing
Further information: Digital signal processing

    Adaptive-additive algorithm (AA algorithm): find the spatial frequency phase of an observed wave source
    Discrete Fourier transform: determines the frequencies contained in a (segment of a) signal
        Bluestein's FFT algorithm
        Bruun's FFT algorithm
        Cooley–Tukey FFT algorithm
        Fast Fourier transform
        Prime-factor FFT algorithm
        Rader's FFT algorithm
    Fast folding algorithm: an efficient algorithm for the detection of approximately periodic events within time series data
    Gerchberg–Saxton algorithm: Phase retrieval algorithm for optical planes
    Goertzel algorithm: identify a particular frequency component in a signal. Can be used for DTMF digit decoding.
    Karplus-Strong string synthesis: physical modelling synthesis to simulate the sound of a hammered or plucked string or some types of percussion

Image processing
Further information: Digital image processing

    Contrast Enhancement
        Histogram equalization: use histogram to improve image contrast
        Adaptive histogram equalization: histogram equalization which adapts to local changes in contrast
    Connected-component labeling: find and label disjoint regions
    Dithering and half-toning
        Error diffusion
        Floyd–Steinberg dithering
        Ordered dithering
        Riemersma dithering
    Elser difference-map algorithm: a search algorithm for general constraint satisfaction problems. Originally used for X-Ray diffraction microscopy
    Feature detection
        Canny edge detector: detect a wide range of edges in images
        Generalised Hough transform
        Hough transform
        Marr–Hildreth algorithm: an early edge detection algorithm
        SIFT (Scale-invariant feature transform): is an algorithm to detect and describe local features in images.
        SURF (Speeded Up Robust Features): is a robust local feature detector, first presented by Herbert Bay et al. in 2006, that can be used in computer vision tasks like object recognition or 3D reconstruction. It is partly inspired by the SIFT descriptor. The standard version of SURF is several times faster than SIFT and claimed by its authors to be more robust against different image transformations than SIFT.[2][3]
    Richardson–Lucy deconvolution: image de-blurring algorithm
    Blind deconvolution: image de-blurring algorithm when point spread function is unknown.
    Median filtering
    Seam carving: content-aware image resizing algorithm
    Segmentation: partition a digital image into two or more regions
        GrowCut algorithm: an interactive segmentation algorithm
        Random walker algorithm
        Region growing
        Watershed transformation: a class of algorithms based on the watershed analogy

Software engineering
Further information: Software engineering

    Cache algorithms
    CHS conversion: converting between disk addressing systems
    Double dabble: Convert binary numbers to BCD
    Hash Function: convert a large, possibly variable-sized amount of data into a small datum, usually a single integer that may serve as an index into an array
        Fowler–Noll–Vo hash function: fast with low collision rate
        Pearson hashing: computes 8 bit value only, optimized for 8 bit computers
        Zobrist hashing: used in the implementation of transposition tables
    Unicode Collation Algorithm
    Xor swap algorithm: swaps the values of two variables without using a buffer

Database algorithms
Further information: Database

    Algorithms for Recovery and Isolation Exploiting Semantics (ARIES): transaction recovery
    Join algorithms
        Block nested loop
        Hash join
        Nested loop join
        Sort-Merge Join

Distributed systems algorithms
Further information: Distributed algorithm and Distributed systems

    Bully algorithm: a method for dynamically selecting a coordinator
    Clock synchronization
        Berkeley algorithm
        Cristian's algorithm
        Intersection algorithm
        Marzullo's algorithm
    Detection of Process Termination
        Dijkstra-Scholten algorithm
        Huang's algorithm
    Lamport ordering: a partial ordering of events based on the happened-before relation
    Mutual exclusion
        Lamport's Distributed Mutual Exclusion Algorithm
        Naimi-Trehel's log(n) Algorithm
        Maekawa's Algorithm
        Raymond's Algorithm
        Ricart-Agrawala Algorithm
    Paxos algorithm: a family of protocols for solving consensus in a network of unreliable processors
    Raft (computer science): a consensus algorithm designed as an alternative to Paxos
    Snapshot algorithm: record a consistent global state for an asynchronous system
        Chandy-Lamport algorithm
    Vector clocks: generate a partial ordering of events in a distributed system and detect causality violations

Memory allocation and deallocation algorithms

    Buddy memory allocation: Algorithm to allocate memory such that fragmentation is less.
    Garbage collectors
        Cheney's algorithm: An improvement on the Semi-space collector
        Generational garbage collector: Fast garbage collectors that segregate memory by age
        Mark-compact algorithm: a combination of the mark-sweep algorithm and Cheney's copying algorithm
        Mark and sweep
        Semi-space collector: An early copying collector
    Reference counting

Networking
Further information: Network scheduler

    Karn's algorithm: addresses the problem of getting accurate estimates of the round-trip time for messages when using TCP
    Luleå algorithm: a technique for storing and searching internet routing tables efficiently
    Network congestion
        Exponential backoff
        Nagle's algorithm: improve the efficiency of TCP/IP networks by coalescing packets
        Truncated binary exponential backoff

Operating systems algorithms
Further information: Operating systems

    Banker's algorithm: Algorithm used for deadlock avoidance.
    Page replacement algorithms: Selecting the victim page under low memory conditions.
        Adaptive replacement cache: better performance than LRU
        Clock with Adaptive Replacement (CAR): is a page replacement algorithm that has performance comparable to Adaptive replacement cache

Process synchronization
Further information: Process synchronization
Further information: Process scheduler

    Dekker's algorithm
    Lamport's Bakery algorithm
    Peterson's algorithm

Scheduling
Further information: Scheduling (computing)

    Earliest deadline first scheduling
    Fair-share scheduling
    Least slack time scheduling
    List scheduling
    Multi level feedback queue
    Rate-monotonic scheduling
    Round-robin scheduling
    Shortest job next
    Shortest remaining time
    Top-nodes algorithm: resource calendar management

I/O scheduling
Further information: I/O scheduling
[icon]	
This section needs expansion. You can help by adding to it. (July 2017)
Disk scheduling

    Elevator algorithm: Disk scheduling algorithm that works like an elevator.
    Shortest seek first: Disk scheduling algorithm to reduce seek time.
    
    
    
        absolute performance guarantee
        abstract data type (ADT)
        (a,b)-tree
        accepting state
        Ackermann's function
        active data structure
        acyclic directed graph
        adaptive heap sort
        adaptive Huffman coding
        adaptive k-d tree
        adaptive sort
        address-calculation sort
        adjacency list representation
        adjacency matrix representation
        adversary
        algorithm
        algorithm BSTW
        algorithm FGK
        algorithmic efficiency
        algorithmically solvable
        algorithm V
        all pairs shortest path
        alphabet
        Alpha Skip Search algorithm
        alternating path
        alternating Turing machine
        alternation
        American flag sort
        amortized cost
        ancestor
        and
        American National Standards Institute (ANSI)
        antichain
        antisymmetric relation
        AP
        Apostolico–Crochemore
        Apostolico–Giancarlo algorithm
        approximate string matching
        approximation algorithm
        arborescence
        arithmetic coding
        array
        array index
        array merging
        array search
        articulation point
        A* search algorithm
        assignment problem
        association list
        associative
        associative array
        asymptotically tight bound
        asymptotic bound
        asymptotic lower bound
        asymptotic space complexity
        asymptotic time complexity
        asymptotic upper bound
        augmenting path
        automaton
        average case
        average-case cost
        AVL tree
        axiomatic semantics
    
    B
    
        backtracking
        bag
        Baillie-PSW primality test
        balanced binary search tree
        balanced binary tree
        balanced k-way merge sort
        balanced merge sort
        balanced multiway merge
        balanced multiway tree
        balanced quicksort
        balanced tree
        balanced two-way merge sort
        BANG file
        Batcher sort
        Baum Welch algorithm
        BB α tree
        BDD
        BD-tree
        Bellman–Ford algorithm
        Benford's law
        best case
        best-case cost
        best-first search
        biconnected component
        biconnected graph
        bidirectional bubble sort
        big-O notation
        binary function
        binary GCD algorithm
        binary heap
        binary insertion sort
        binary knapsack problem
        binary priority queue
        binary relation
        binary search
        binary search tree
        binary tree
        binary tree representation of trees
        bingo sort
        binomial heap
        binomial tree
        bin packing problem
        bin sort
        bintree
        bipartite graph
        bipartite matching
        bisector
        bitonic sort
        bit vector
        Bk tree
        block
        block addressing index
        blocking flow
        block search
        Bloom filter
        blossom (graph theory)
        bogosort
        boogol
        boolean
        boolean expression
        boolean function
        bottleneck traveling salesman
        bottom-up tree automaton
        boundary-based representation
        bounded error probability in polynomial time
        bounded queue
        bounded stack
        Bounding volume hierarchy, also referred to as bounding volume tree (BV-tree, BVT)
        Boyer–Moore string search algorithm
        Boyer–Moore–Horspool algorithm
        bozo sort
        B+ tree
        BPP (complexity)
        Bradford's law
        branch (as in control flow)
        branch (as in revision control)
        branch and bound
        breadth-first search
        Bresenham's algorithm
        brick sort
        bridge
        British Museum algorithm
        brute force attack
        brute force search
        brute force string search
        brute force string search with mismatches
        BSP-tree
        B*-tree
        B-tree
        bubble sort
        bucket
        bucket array
        bucketing method
        bucket sort
        bucket trie
        buddy system
        buddy tree
        build-heap
        Burrows–Wheeler transform (BWT)
        busy beaver
        Byzantine generals
    
    C
    
        cactus stack
        Calculus of Communicating Systems (CCS)
        calendar queue
        candidate consistency testing
        candidate verification
        canonical complexity class
        capacitated facility location
        capacity
        capacity constraint
        cartesian tree
        cascade merge sort
        caverphone
        Cayley–Purser algorithm
        C curve
        cell probe model
        cell tree
        cellular automaton
        centroid
        certificate
        chain (order theory)
        chaining (algorithm)
        child
        Chinese postman problem
        Chinese remainder theorem
        Christofides algorithm
        Christofides heuristic
        chromatic index
        chromatic number
        Church–Turing thesis
        circuit
        circuit complexity
        circuit value problem
        circular list
        circular queue
        clique
        clique problem
        clustering (see hash table)
        clustering free
        coalesced hashing
        coarsening
        cocktail shaker sort
        codeword
        coding tree
        collective recursion
        collision
        collision resolution scheme
        Colussi
        combination
        comb sort
        Communicating Sequential Processes
        commutative
        compact DAWG
        compact trie
        comparison sort
        competitive analysis
        competitive ratio
        complement
        complete binary tree
        complete graph
        completely connected graph
        complete tree
        complexity
        complexity class
        computable
        concave function
        concurrent flow
        concurrent read, concurrent write
        concurrent read, exclusive write
        configuration
        confluently persistent data structure
        conjunction
        connected components
        connected graph
        co-NP
        constant function
        continuous knapsack problem
        Cook reduction
        Cook's theorem
        counting sort
        covering
        CRCW
        Crew (algorithm)
        critical path problem
        CSP (communicating sequential processes)
        CSP (constraint satisfaction problem)
        CTL
        cuckoo hashing
        cut (graph theory)
        cut (logic programming)
        cutting plane
        cutting stock problem
        cutting theorem
        cut vertex
        cycle sort
        cyclic redundancy check (CRC)
    
    D
    
        D-adjacent
        DAG shortest paths
        Damerau–Levenshtein distance
        data structure
        decidable
        decidable language
        decimation
        decision problem
        decision tree
        decomposable searching problem
        degree
        dense graph
        depoissonization
        depth
        depth-first search (DFS)
        deque
        derangement
        descendant (see tree structure)
        deterministic
        deterministic algorithm
        deterministic finite automata string search
        deterministic finite automaton (DFA)
        deterministic finite state machine
        deterministic finite tree automaton
        deterministic pushdown automaton (DPDA)
        deterministic tree automaton
        Deutsch–Jozsa algorithm
        DFS forest
        DFTA
        diagonalization argument
        diameter
        dichotomic search
        dictionary (data structure)
        diet (see discrete interval encoding tree below)
        difference (set theory)
        digital search tree
        digital tree
        digraph
        Dijkstra's algorithm
        diminishing increment sort
        dining philosophers
        direct chaining hashing
        directed acyclic graph (DAG)
        directed acyclic word graph (DAWG)
        directed graph
        discrete interval encoding tree
        discrete p-center
        disjoint set
        disjunction
        distributed algorithm
        distributional complexity
        distribution sort
        divide and conquer algorithm
        divide and marriage before conquest
        division method
        Data domain
        don't care
        Doomsday rule
        double-direction bubble sort
        double-ended priority queue
        double hashing
        double left rotation
        Double Metaphone
        double right rotation
        doubly ended queue
        doubly linked list
        Dragon curve
        dual graph
        dual linear program
        dyadic tree
        dynamic array
        dynamic data structure
        dynamic hashing
        dynamic programming
        dynamization transformation
    
    E
    
        edge
        edge coloring
        edge connectivity
        edge crossing
        edge-weighted graph
        edit distance
        edit operation
        edit script
        8 queens
        elastic-bucket trie
        element uniqueness
        end-of-string
        enfilade
        epidemic algorithm
        Euclidean algorithm
        Euclidean distance
        Euclidean Steiner tree
        Euclidean traveling salesman problem
        Euclid's algorithm
        Euler cycle
        Eulerian graph
        Eulerian path
        exact string matching
        EXCELL (extendible cell)
        exchange sort
        exclusive or
        exclusive read, concurrent write (ERCW)
        exclusive read, exclusive write (EREW)
        exhaustive search
        existential state
        expandable hashing
        expander graph
        exponential
        extended binary tree
        extended Euclidean algorithm
        extended k-d tree
        extendible hashing
        external index
        external memory algorithm
        external memory data structure
        external merge
        external merge sort
        external node
        external quicksort
        external radix sort
        external sort
        extrapolation search
        extremal
        extreme point
    
    F
    
        facility location
        factor (see substring)
        factorial
        fast fourier transform (FFT)
        fathoming
        feasible region
        feasible solution
        feedback edge set
        feedback vertex set
        Ferguson–Forcade algorithm
        Fibonacci number
        Fibonacci search
        Fibonacci tree
        Fibonacci heap
        Find
        find kth least element
        finitary tree
        finite Fourier transform (discrete Fourier transform)
        finite state automaton
        finite state machine
        finite state machine minimization
        finite state transducer
        first come, first served
        first-in, first-out (FIFO)
        fixed-grid method
        flash sort
        flow
        flow conservation
        flow function
        flow network
        Floyd–Warshall algorithm
        Ford–Bellman algorithm
        Ford–Fulkerson algorithm
        forest
        forest editing problem
        formal language
        formal methods
        formal verification
        forward index
        fractal
        fractional knapsack problem
        fractional solution
        free edge
        free list
        free tree
        free vertex
        frequency count heuristic
        full array
        full binary tree
        full inverted index
        fully dynamic graph problem
        fully persistent data structure
        fully polynomial approximation scheme
        function (programming)
        function (mathematics)
        functional data structure
    
    G
    
        Galil–Giancarlo
        Galil–Seiferas
        gamma function
        GBD-tree
        geometric optimization problem
        global optimum
        gnome sort
        goobi
        graph
        graph coloring
        graph concentration
        graph drawing
        graph isomorphism
        graph partition
        Gray code
        greatest common divisor (GCD)
        greedy algorithm
        greedy heuristic
        grid drawing
        grid file
        Grover's algorithm
    
    H
    
        halting problem
        Hamiltonian cycle
        Hamiltonian path
        Hamming distance
        Harter–Highway dragon
        hash function
        hash heap
        hash table
        hash table delete
        Hausdorff distance
        hB-tree
        head
        heap
        heapify
        heap property
        heapsort
        heaviest common subsequence
        height
        height-balanced binary search tree
        height-balanced tree
        heuristic
        hidden Markov model
        highest common factor
        Hilbert curve
        histogram sort
        homeomorphic
        horizontal visibility map
        Huffman encoding
        Hungarian algorithm
        hybrid algorithm
        hyperedge
        hypergraph
    
    I
    
        Identity function
        ideal merge
        implication
        implies
        in-branching
        inclusion–exclusion principle
        inclusive or
        incompressible string
        incremental algorithm
        in-degree
        independent set (graph theory)
        index file
        information theoretic bound
        in-order traversal
        in-place sort
        insertion sort
        instantaneous description
        integer linear program
        integer multi-commodity flow
        integer polyhedron
        interactive proof system
        Interface_(computing)
        interior-based representation
        internal node
        internal sort
        interpolation search
        interpolation-sequential search
        interpolation sort
        intersection (set theory)
        interval tree
        intractable
        introsort
        introspective sort
        inverse Ackermann function
        inverted file index
        inverted index
        irreflexive
        isomorphic
        iteration
    
    J
    
        Jaro–Winkler distance
        Johnson's algorithm
        Johnson–Trotter algorithm
        jump list
        jump search
    
    K
    
        Karmarkar's algorithm
        Karnaugh map
        Karp–Rabin string search algorithm
        Karp reduction
        k-ary heap
        k-ary Huffman encoding
        k-ary tree
        k-clustering
        k-coloring
        k-connected graph
        k-d-B-tree
        k-dimensional
        K-dominant match
        k-d tree
        key
        KMP
        KmpSkip Search
        knapsack problem
        knight's tour
        Knuth–Morris–Pratt algorithm
        Königsberg bridges problem
        Kolmogorov complexity
        Kraft's inequality
        Kripke structure
        Kruskal's algorithm
        kth order Fibonacci numbers
        kth shortest path
        kth smallest element
        KV diagram
        k-way merge
        k-way merge sort
        k-way tree
    
    L
    
        labeled graph
        language
        last-in, first-out (LIFO)
        Las Vegas algorithm
        lattice (group)
        layered graph
        LCS
        leaf
        least common multiple (LCM)
        leftist tree
        left rotation
        Left-child right-sibling binary tree also termed first-child next-sibling binary tree, doubly chained tree, or filial-heir chain
        Lempel–Ziv–Welch (LZW)
        level-order traversal
        Levenshtein distance
        lexicographical order
        linear
        linear congruential generator
        linear hash
        linear insertion sort
        linear order
        linear probing
        linear probing sort
        linear product
        linear program
        linear quadtree
        linear search
        link
        linked list
        list
        list contraction
        little-o notation
        Lm distance
        load factor (computer science)
        local alignment
        local optimum
        logarithm, logarithmic scale
        longest common subsequence
        longest common substring
        Lotka's law
        lower bound
        lower triangular matrix
        lowest common ancestor
        l-reduction
    
    M
    
        Malhotra–Kumar–Maheshwari blocking flow (ru.)
        Manhattan distance
        many-one reduction
        Markov chain
        marriage problem (see assignment problem)
        Master theorem (analysis of algorithms)
        matched edge
        matched vertex
        matching (graph theory)
        matrix
        matrix-chain multiplication problem
        max-heap property
        maximal independent set
        maximally connected component
        Maximal Shift
        maximum bipartite matching
        maximum-flow problem
        MAX-SNP
        Mealy machine
        mean
        median
        meld (data structures)
        memoization
        merge algorithm
        merge sort
        Merkle tree
        meromorphic function
        metaheuristic
        metaphone
        midrange
        Miller–Rabin primality test
        min-heap property
        minimal perfect hashing
        minimum bounding box (MBB)
        minimum cut
        minimum path cover
        minimum spanning tree
        minimum vertex cut
        mixed integer linear program
        mode
        model checking
        model of computation
        moderately exponential
        MODIFIND
        monotone priority queue
        monotonically decreasing
        monotonically increasing
        Monte Carlo algorithm
        Moore machine
        Morris-Pratt
        move (finite-state machine transition)
        move-to-front heuristic
        move-to-root heuristic
        multi-commodity flow
        multigraph
        multilayer grid file
        multiplication method
        multiprefix
        multiprocessor model
        multiset
        multi suffix tree
        multiway decision
        multiway merge
        multiway search tree
        multiway tree
        Munkres' assignment algorithm
    
    N
    
        naive string search
        nand
        n-ary function
        NC
        NC many-one reducibility
        nearest neighbor search
        negation
        network flow (see flow network)
        network flow problem
        next state
        NIST
        node
        nonbalanced merge
        nonbalanced merge sort
        nondeterministic
        nondeterministic algorithm
        nondeterministic finite automaton
        nondeterministic finite state machine (NFA)
        nondeterministic finite tree automaton (NFTA)
        nondeterministic polynomial time
        nondeterministic tree automaton
        nondeterministic Turing machine
        nonterminal node
        nor
        not
        Not So Naive
        NP
        NP-complete
        NP-complete language
        NP-hard
        n queens
        nullary function
        null tree
        New York State Identification and Intelligence System (NYSIIS)
    
    O
    
        objective function
        occurrence
        octree
        offline algorithm
        offset (computer science)
        omega
        omicron
        one-based indexing
        one-dimensional
        online algorithm
        open addressing
        optimal
        optimal cost
        optimal hashing
        optimal merge
        optimal mismatch
        optimal polygon triangulation problem
        optimal polyphase merge
        optimal polyphase merge sort
        optimal solution
        optimal triangulation problem
        optimal value
        optimization problem
        or
        oracle set
        oracle tape
        oracle Turing machine
        Orders of approximation
        ordered array
        ordered binary decision diagram (OBDD)
        ordered linked list
        ordered tree
        order preserving hash
        order preserving minimal perfect hashing
        oriented acyclic graph
        oriented graph
        oriented tree
        orthogonal drawing
        orthogonal lists
        orthogonally convex rectilinear polygon
        oscillating merge sort
        out-branching
        out-degree
        overlapping subproblems
    
    P
    
        packing (see set packing)
        padding argument
        pagoda
        pairing heap
        PAM (point access method)
        parallel computation thesis
        parallel prefix computation
        parallel random-access machine (PRAM)
        parametric searching
        parent
        partial function
        partially decidable problem
        partially dynamic graph problem
        partially ordered set
        partially persistent data structure
        partial order
        partial recursive function
        partition (set theory)
        passive data structure
        patience sorting
        path (graph theory)
        path cover
        path system problem
        Patricia tree
        pattern
        pattern element
        P-complete
        PCP
        Peano curve
        Pearson's hashing
        perfect binary tree
        perfect hashing
        perfect k-ary tree
        perfect matching
        perfect shuffle
        performance guarantee
        performance ratio
        permutation
        persistent data structure
        phonetic coding
        pile (data structure)
        pipelined divide and conquer
        planar graph
        planarization
        planar straight-line graph
        PLOP-hashing
        point access method
        pointer jumping
        pointer machine
        poissonization
        polychotomy
        polyhedron
        polylogarithmic
        polynomial
        polynomial-time approximation scheme (PTAS)
        polynomial hierarchy
        polynomial time
        polynomial-time Church–Turing thesis
        polynomial-time reduction
        polyphase merge
        polyphase merge sort
        polytope
        poset
        postfix traversal
        Post machine (see Post–Turing machine)
        postman's sort
        postorder traversal
        Post correspondence problem
        potential function (see potential method)
        predicate
        prefix
        prefix code
        prefix computation
        prefix sum
        prefix traversal
        preorder traversal
        primary clustering
        primitive recursive
        Prim's algorithm
        principle of optimality
        priority queue
        prisoner's dilemma
        PRNG
        probabilistic algorithm
        probabilistically checkable proof
        probabilistic Turing machine
        probe sequence
        Procedure (computer science)
        process algebra
        proper (see proper subset)
        proper binary tree
        proper coloring
        proper subset
        property list
        prune and search
        pseudorandom number generator
        pth order Fibonacci numbers
        P-tree
        purely functional language
        pushdown automaton (PDA)
        pushdown transducer
        p-way merge sort
    
    Q
    
        qm sort
        qsort
        quadratic probing
        quadtree
        quadtree complexity theorem
        quad trie
        quantum computation
        queue
        quicksort
    
    R
    
        Rabin–Karp string search algorithm
        radix quicksort
        radix sort
        ragged matrix
        Raita algorithm
        random access machine
        random number generation
        randomization
        randomized algorithm
        randomized binary search tree
        randomized complexity
        randomized polynomial time
        randomized rounding
        randomized search tree
        Randomized-Select
        random number generator
        random sampling
        range (function)
        range sort
        Rank (graph theory)
        Ratcliff/Obershelp pattern recognition
        reachable
        rebalance
        recognizer
        rectangular matrix
        rectilinear
        rectilinear Steiner tree
        recurrence equations
        recurrence relation
        recursion
        recursion termination
        recursion tree
        recursive (computer science)
        recursive data structure
        recursive doubling
        recursive language
        recursively enumerable language
        recursively solvable
        red-black tree
        reduced basis
        reduced digraph
        reduced ordered binary decision diagram (ROBDD)
        reduction
        reflexive relation
        regular decomposition
        rehashing
        relation (mathematics)
        relational structure
        relative performance guarantee
        relaxation
        relaxed balance
        rescalable
        restricted universe sort
        result cache
        Reverse Colussi
        Reverse Factor
        R-file
        Rice's method
        right rotation
        right-threaded tree
        root
        root balance
        rooted tree
        rotate left
        rotate right
        rotation
        rough graph
        RP
        R+-tree
        R*-tree
        R-tree
        run time
    
    S
    
        saguaro stack
        saturated edge
        SBB tree
        scan
        scapegoat tree
        search algorithm
        search tree
        search tree property
        secant search
        secondary clustering
        memory segment
        select algorithm
        select and partition
        selection problem
        selection sort
        select kth element
        select mode
        self-loop
        self-organizing heuristic
        self-organizing list
        self-organizing sequential search
        semidefinite programming
        separate chaining hashing
        separation theorem[disambiguation needed]
        sequential search
        set
        set cover
        set packing
        shadow heap
        shadow merge
        shadow merge insert
        shaker sort
        Shannon–Fano coding
        shared memory
        Shell sort
        Shift-Or
        Shor's algorithm
        shortcutting
        shortest common supersequence
        shortest common superstring
        shortest path
        shortest spanning tree
        shuffle
        shuffle sort
        sibling
        Sierpiński curve
        Sierpinski triangle
        sieve of Eratosthenes
        sift up
        signature
        Simon's algorithm
        simple merge
        simple path
        simple uniform hashing
        simplex communication
        simulated annealing
        simulation theorem
        single-destination shortest-path problem
        single-pair shortest-path problem
        single program multiple data
        single-source shortest-path problem
        singly linked list
        singularity analysis
        sink
        sinking sort
        skd-tree
        skew symmetry
        skip list
        skip search
        slope selection
        Smith algorithm
        Smith–Waterman algorithm
        smoothsort
        solvable problem
        sort algorithm
        sorted array
        sorted list
        sort in place
        sort merge
        soundex
        space-constructible function
        spanning tree
        sparse graph
        sparse matrix
        sparsification
        sparsity
        spatial access method
        spectral test
        splay tree
        SPMD
        square matrix
        square root
        SST (shortest spanning tree)
        stable
        stack (data structure)
        stack tree
        star-shaped polygon
        start state
        state
        state machine
        state transition
        static data structure
        static Huffman encoding
        s-t cut
        st-digraph
        Steiner minimum tree
        Steiner point
        Steiner ratio
        Steiner tree
        Steiner vertex
        Steinhaus–Johnson–Trotter algorithm
        Stirling's approximation
        Stirling's formula
        stooge sort
        straight-line drawing
        strand sort
        strictly decreasing
        strictly increasing
        strictly lower triangular matrix
        strictly upper triangular matrix
        string
        string editing problem
        string matching
        string matching on ordered alphabets
        string matching with errors
        string matching with mismatches
        string searching
        strip packing
        strongly connected component
        strongly connected graph
        strongly NP-hard
        subadditive ergodic theorem
        subgraph isomorphism
        sublinear time algorithm
        subsequence
        subset
        substring
        subtree
        suffix
        suffix array
        suffix automaton
        suffix tree
        superimposed code
        superset
        supersink
        supersource
        symmetric relation
        symmetrically linked list
        symmetric binary B-tree
        symmetric set difference
        symmetry breaking
        symmetric min max heap
    
    T
    
        tail
        tail recursion
        target
        temporal logic
        terminal (see Steiner tree)
        terminal node
        ternary search
        ternary search tree (TST)
        text searching
        theta
        threaded binary tree
        threaded tree
        three-dimensional
        three-way merge sort
        three-way radix quicksort
        time-constructible function
        time/space complexity
        top-down radix sort
        top-down tree automaton
        top-node
        topological order
        topological sort
        topology tree
        total function
        totally decidable language
        totally decidable problem
        totally undecidable problem
        total order
        tour
        tournament
        towers of Hanoi
        tractable problem
        transducer
        transition (see finite-state machine)
        transition function (of a finite-state machine or Turing machine)
        transitive relation
        transitive closure
        transitive reduction
        transpose sequential search
        travelling salesman problem (TSP)
        treap
        tree
        tree automaton
        tree contraction
        tree editing problem
        tree sort
        tree transducer
        tree traversal
        triangle inequality
        triconnected graph
        trie
        trinary function
        tripartition
        Turbo-BM
        Turbo Reverse Factor
        Turing machine
        Turing reduction
        Turing transducer
        twin grid file
        two-dimensional
        two-level grid file
        2-3-4 tree
        2-3 tree
        Two Way algorithm
        two-way linked list
        two-way merge sort
    
    U
    
        unary function
        unbounded knapsack problem (UKP)
        uncomputable function
        uncomputable problem
        undecidable language
        undecidable problem
        undirected graph
        uniform circuit complexity
        uniform circuit family
        uniform hashing
        uniform matrix
        union
        union of automata
        universal hashing
        universal state
        universal Turing machine
        universe
        unsolvable problem
        unsorted list
        upper triangular matrix
    
    V
    
        van Emde Boas priority queue
        vehicle routing problem
        Veitch diagram
        Venn diagram
        vertex
        vertex coloring
        vertex connectivity
        vertex cover
        vertical visibility map
        virtual hashing
        visibility map
        visible (geometry)
        Viterbi algorithm
        VP-tree
        VRP (vehicle routing problem)
    
    W
    
        walk
        weak cluster
        weak-heap
        weak-heap sort
        weight-balanced tree
        weighted, directed graph
        weighted graph
        window
        witness
        work-depth model
        work-efficient
        work-preserving
        worst case
        worst-case cost
        worst-case minimum access
    
    X
    
        xor
    
    Y
    
        Yule–Simon distribution
    
    Z
    
        Zeller's congruence
        0-ary function
        0-based indexing
        0/1 knapsack problem
        Zhu–Takaoka string matching algorithm
        Zipfian distribution
        Zipf's law
        Zipper (data structure)
        ZPP