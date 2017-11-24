#include <iostream>
#include "stdNNO.h"

using namespace std;
using namespace NeuralNet;


int main()
{
   int NumOfLayers;
	cout << "Input number of layers:" << endl;
	cin >> NumOfLayers;
	int *sizes = new int[NumOfLayers];
	for (int i = 0; i < NumOfLayers; i++)
	{
		cout << "Input size of layer " << i + 1 << ": " << endl;
		cin >> sizes[i];
	} 
	NeuralNerwork a(NumOfLayers, sizes);
	double *input = new double[a.GetSizeOfLayer(1)];
	double *output = new double[a.GetSizeOfLayer(a.GetNumberOfLayers())];
	for (int i = 0; i <1000; i++)
	{
		for (int randvar = 0; randvar < 1; randvar++) {
			for (int j = 0; j < a.GetSizeOfLayer(1); j++)
			{
				if (j != randvar)
				{
					input[j] = 0;
				}
				else
				{
					input[j] = randvar / 10.0;
				}
				cout << input[j] << " ";
			}
			cout << endl;
					output[0] = input[randvar] + 0.01;
					cout << output[0] << endl;
			a.BackPropagationTeaching(input, output, 5);
			a.GetOutputOfTheLayer(a.GetNumberOfLayers());
		}
		system("pause");
	}
	for (int j = 0; j < a.GetSizeOfLayer(1); j++)
	{
		cout << "Input: " << endl;
		cin >> input[j];
	}
	a.RunNeuralNet(input);
	a.GetOutputOfTheLayer(a.GetNumberOfLayers());
	system("pause");
	return 0;
}