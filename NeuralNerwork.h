#pragma once
#include "LayerOfNeurons.h"

using namespace NeuralNet;

const int MAX_NUMBER_OF_LAYERS = 20;

class NeuralNerwork
{
private:
	LayerOfNeurons NNet[MAX_NUMBER_OF_LAYERS];
	int NumberOfLayers;
public:
	NeuralNerwork();
	NeuralNerwork(int NumOfLayers, int *sizes);
	void SetNumberOfLayers(int number);
	int GetNumberOfLayers();
	void ReturnNeuralNetFromMemory();
	void CreateNeuralNet(int NumOfLayers, int *sizes);
	void RunNeuralNet(double *input);
	void GetOutputOfTheLayer(int number);
	int GetSizeOfLayer(int number);
	void ShowNeuralNetwork();
	void BackPropagationTeaching(double* TrainingSet, double* IdealOutput, double LearningRate);
	LayerOfNeurons ReturnLayer(int i);
	~NeuralNerwork();
};



