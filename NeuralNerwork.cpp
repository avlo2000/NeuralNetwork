#include "NeuralNerwork.h"
#include <vector>
using namespace std;
using namespace NeuralNet;
double sigmoid1(double x)
{
	double exp_value;
	double return_value;
	exp_value = exp((double)-x);
	return_value = 1 / (1 + exp_value);

	return return_value;
}

double sigmoid_derivative1(double x)
{
	double return_value;
	return_value = sigmoid1(x)*(1 - sigmoid1(x));
	return return_value;
}

NeuralNerwork::NeuralNerwork()
{
	ReturnNeuralNetFromMemory();
}

NeuralNerwork::NeuralNerwork(int NumOfLayers, int *sizes)
{
	CreateNeuralNet(NumOfLayers, sizes);
}

void NeuralNerwork::SetNumberOfLayers(int number)
{
	NumberOfLayers = number;
}

int NeuralNerwork::GetNumberOfLayers()
{
	return NumberOfLayers;
}

void NeuralNerwork::ReturnNeuralNetFromMemory()
{
	cout << endl << endl << "      ...RETURNING NEURAL NETWORK FROM MEMORY FILES..." << endl << endl;
	ifstream fin1("SizesOfLayers.txt", ios_base::in);
	ifstream fin2("NumberOfLayers.txt", ios_base::in);
	int buffer;
	fin2 >> buffer;
	SetNumberOfLayers(buffer);
	for (int i = 0; i < NumberOfLayers; i++)
	{
		fin1 >> buffer;
		NNet[i].SetSizeOfLayer(buffer);
		if (i != 0)
		{
			NNet[i].SetNumberOfLayer(i);
			NNet[i].SetSynapsesForLayer(NNet[i-1]);
		}
	}
}

void NeuralNerwork::CreateNeuralNet(int NumOfLayers, int *sizes)
{
	int Buffer;
	double randvar;
	cout << endl << endl << "      ...CREATING NEURAL NETWORK..." << endl << endl;
	SetNumberOfLayers(NumOfLayers);
	ofstream fout1("SizesOfLayers.txt", ios_base::out | ios_base::trunc);
	ofstream fout2("NumberOfLayers.txt", ios_base::out | ios_base::trunc);
	fout2 << NumOfLayers; // remembering
	for (int i = 0; i < NumOfLayers; ++i)
	{
		NNet[i].SetSizeOfLayer(sizes[i]);
		fout1 << sizes[i] << endl; // remembering
		if (i != 0)
		{
			NNet[i].SetNumberOfLayer(i);
			randvar = 0.5;
			NNet[i].SetSynapsesForLayer(NNet[i - 1], randvar);
		}
	}
	ShowNeuralNetwork();
	fout1.close();
	fout2.close();
}

void NeuralNerwork::RunNeuralNet(double *input)
{
	//std::cout << endl << endl << "      ...RUNNING NEURAL NETWORK..." << endl << endl;
	double inputer;
	for (int i = 0; i < NumberOfLayers; i++)
	{
		if (i == 0)
		{
			NNet[i].SetInputOfLayer(true, input);
		}
		else
		{
			NNet[i].SetInputOfLayer(NNet[i-1]);
		}
	}
}

void NeuralNerwork::GetOutputOfTheLayer(int number)
{
	cout << endl << endl << "      ...GETTING OUTPUT FROM LAYER NUMBER "<< number << "..." << endl << endl;
	for (int i = 0; i < NNet[number - 1].GetSizeOfLayer(); i++)
	{
	     cout << "Output number " << i + 1 << "is: " << NNet[number - 1].GetOutputOfNeuron(i) << endl;
	}
}

int NeuralNerwork::GetSizeOfLayer(int number)
{
	return NNet[number-1].GetSizeOfLayer();
}

LayerOfNeurons NeuralNerwork::ReturnLayer(int i)
{
	return NNet[i];
}

void NeuralNerwork::ShowNeuralNetwork()
{
	for (int i = 1; i < NumberOfLayers; i++)
	{
		cout << "Matrix number " << i  << ":" << endl;
		NNet[i].GetSynapsesForLayer().OutputSynapsMatrix(NNet[i].GetSizeOfLayer(), NNet[i - 1].GetSizeOfLayer());
		cout << endl;
	}
}

void NeuralNerwork::BackPropagationTeaching(double* TrainingSet, double* IdealOutput, double LearningRate)
{
	RunNeuralNet(TrainingSet);
	double error;
	double weight_delta[MAX_SIZE_OF_NeuralNet][MAX_SIZE_OF_LAYER];
	for (int j = 0; j < NNet[NumberOfLayers - 1].GetSizeOfLayer(); j++)
	{
		error = NNet[NumberOfLayers - 1].GetOutputOfNeuron(j) - IdealOutput[j];
		weight_delta[NumberOfLayers - 1][j] = NNet[NumberOfLayers - 1].ReturnNeuron(j).GetDelta(error);
	}
	for (int i = NumberOfLayers - 1; i > 0; i--)
	{
		if (i == NumberOfLayers - 1)
		{
			for (int k = 0; k < NNet[i - 1].GetSizeOfLayer(); k++)
			{
				for (int d = 0; d < NNet[i].GetSizeOfLayer(); d++)
				{
					NNet[i].Synapses.SetWeightOfSynapse(d, k, NNet[i].Synapses.GetWeightOfSynapse(d, k) - LearningRate * weight_delta[i][d] * NNet[i].GetOutputOfNeuron(d));
				}
			}
		}
		
		for (int j = 0; j < NNet[i].GetSizeOfLayer(); j++)
		{
			if (i != NumberOfLayers - 1)
			{
				error = 0;
				for (int k = 0; k < NNet[i + 1].GetSizeOfLayer(); k++)
				{
					error += NNet[i + 1].Synapses.GetWeightOfSynapse(k, j) * weight_delta[i + 1][k];
				}
				weight_delta[i][j] = NNet[i].ReturnNeuron(j).GetDelta(error);
			}
			if (j == NNet[i].GetSizeOfLayer() - 1)
			{
				for (int k = 0; k < NNet[i - 1].GetSizeOfLayer(); k++)
				{
					for (int d = 0; d < NNet[i].GetSizeOfLayer(); d++)
					{
						NNet[i].Synapses.SetWeightOfSynapse(d, k, NNet[i].Synapses.GetWeightOfSynapse(d, k) - LearningRate * weight_delta[i][d] * NNet[i].GetOutputOfNeuron(d)); 						
					}
				}
				NNet[i].Synapses.SetMatrixInMemory(NNet[i].GetSizeOfLayer(), NNet[i - 1].GetSizeOfLayer());
			}
			
		}
		
	}
};

NeuralNerwork::~NeuralNerwork()
{
}
