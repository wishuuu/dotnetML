using System.Globalization;
using IrisClassification;

Console.WriteLine("Hello, World!");

double[] mapData(int i) => i switch
{
    0 => new[] {1d, 0d, 0d},
    1 => new[] {0d, 1d, 0d},
    2 => new[] {0d, 0d, 1d},
    _ => throw new ArgumentException("Invalid index"),
};

double[][] normalizeInputs(double[][] input)
{
    for(int i=0; i<input[0].Length; i++)
    {
        double max = input[0][i];
        double min = input[0][i];
        for(int j=0; j<input.Length; j++)
        {
            if(input[j][i] > max)
            {
                max = input[j][i];
            }
            if(input[j][i] < min)
            {
                min = input[j][i];
            }
        }
        for(int j=0; j<input.Length; j++)
        {
            input[j][i] = (input[j][i] - min) / (max - min);
        }
    }

    return input;
}

bool isPredictionValid(double[] prediction, double[] expected)
{
    int predictionIndex = Array.IndexOf(prediction, prediction.Max());
    int expectedIndex = Array.IndexOf(expected, expected.Max());
    return predictionIndex == expectedIndex;
}

// read Datasets/Iris.csv into variable
using (var reader = new StreamReader("Datasets/Iris.csv"))
{
    var line = reader.ReadLine();
    line = reader.ReadLine();
    
    double[][] inputs = new double[150][];
    double[][] outputs = new double[150][];

    var i = 0;
    while (line != null)
    {
        var values = line.Split(',');
        
        inputs[i] = new double[4];

        for (var j = 1; j < 5; j++)
        {
            inputs[i][j-1] = double.Parse(values[j], CultureInfo.InvariantCulture);
        }

        outputs[i] = mapData(int.Parse(values[5], CultureInfo.InvariantCulture));
        
        line = reader.ReadLine();
        i++;
    }
    
    // randomize the order of inputs and outputs
    var random = new Random();
    for (i = 0; i < inputs.Length; i++)
    {
        var j = random.Next(inputs.Length);
        var temp = inputs[i];
        inputs[i] = inputs[j];
        inputs[j] = temp;
        
        temp = outputs[i];
        outputs[i] = outputs[j];
        outputs[j] = temp;
    }

    inputs = normalizeInputs(inputs);


    IrisClassificator classifier = new IrisClassificator(inputs, outputs);
    var errors = classifier.Train(40);

    var total = 0;
    var correct = 0;
    for (i=0; i<inputs.Length; i++)
    {
        var prediction = classifier.Predict(inputs[i]);
        if(isPredictionValid(prediction, outputs[i]))
        {
            correct++;
        }
        total++;
    }
    
    if (total==0)
    {
        Console.WriteLine("No data");
    }
    else
    {
        Console.WriteLine($"Accuracy: {(double)correct/total}");
    }
}