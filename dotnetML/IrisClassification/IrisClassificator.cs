using AForge.Neuro;
using AForge.Neuro.Learning;

namespace IrisClassification;

public class IrisClassificator
{
    private double[][] _input;
    private double[][] _output;

    private ActivationNetwork _network;
    private BackPropagationLearning _teacher;

    public IrisClassificator(double[][] input, double[][] output, double learningRate = 0.1)
    {
        _input = input;
        _output = output;
        _network = new ActivationNetwork(new BipolarSigmoidFunction(), 4, 5, 3);
        _network.Randomize();
        _teacher = new BackPropagationLearning(_network)
        {
            LearningRate = learningRate
        };
    }

    public void FetchData(double[][] input, double[][] output)
    {
        _input = input;
        _output = output;
    }

    public double[] Train(int epochs = 3)
    {
        double[] errors = new double[epochs];
        Console.WriteLine("Learning in progress...");
        for (int i = 0; i < epochs; i++)
        {
            Console.WriteLine("Epoch #" + (i + 1));
            errors[i] = _teacher.RunEpoch(_input, _output);
        }

        return errors;
    }
    
    public double[] Predict(double[] input)
    {
        return _network.Compute(input);
    }
}