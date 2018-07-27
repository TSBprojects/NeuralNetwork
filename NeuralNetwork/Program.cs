using System;

namespace NeuralNetwork
{
    class Program
    {
        static void Main(string[] args)
        {
            NeuralNet net = new NeuralNet(new int[] { 2, 3, 1 });

            double[][][] trainData = net.GetTrainData("trainSet.txt");

            net.OpenNet("1.xml");
            //net.Training(trainData);
            //net.SaveNet("1.xml");

            Console.WriteLine("in: 0 0\nout: {0}\n\nin: 1 1\nout: {1}\n\nin: 0 1\nout: {2}\n\nin: 1 0\nout: {3}\n\n",
                net.Ask(new double[] { 0, 0 })[0],
                net.Ask(new double[] { 1, 1 })[0],
                net.Ask(new double[] { 0, 1 })[0],
                net.Ask(new double[] { 1, 0 })[0]);
            Console.ReadKey();
        }
    }
}
