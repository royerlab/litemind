// C++ Program to find the nth fibonacci number using
// recursion
#include <bits/stdc++.h>
using namespace std;

int fib(int n) {

	// If n is 1 or 0, then return n, works for 0th
  	// and 1st terms
    if (n <= 1)
        return n;

    // Recurrence relation to find the rest of the
  	// terms
    return fib(n - 1) + fib(n - 2);
}

int main() {
    int n = 5;

  	// Finding nth term
    cout << fib(n);
    return 0;
}