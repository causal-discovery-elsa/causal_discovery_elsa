#include <iostream>     // std::cout
#include <fstream>      // std::ifstream
using namespace std;

#include "Hungarian.h"


int main(void)
{
    // please use "-std=c++11" for this initialization of vector.
	// vector< vector<double> > costMatrix = { { 10, 19, 8, 15, 0 }, 
	// 									  { 10, 18, 7, 17, 0 }, 
	// 									  { 13, 16, 9, 14, 0 }, 
	// 									  { 12, 19, 8, 18, 0 } };

	vector< vector<double> > costMatrix;



	ifstream read("matrix.txt");
	int r,c;
	read>>r>>c;


	// int** a = new int*[r];
	// for(int i = 0; i < r; ++i)
	//     a[i] = new int[c];

	double temp;
	for (int i = 0; i < r; i++) {
	    vector<double> row; // Create an empty row
	    for (int j = 0; j < c; j++) {
	    	read>> temp;
	        row.push_back(temp); // Add an element (column) to the row
	    }
	    costMatrix.push_back(row); // Add the row to the main vector
	}			


	HungarianAlgorithm HungAlgo;
	vector<int> assignment;

	double cost = HungAlgo.Solve(costMatrix, assignment);


	ofstream write("matching.txt");
	for ( unsigned int x = 0; x < costMatrix.size(); x++ )
		write << x << "," << assignment[x] << "\n";

	return 0;
}
