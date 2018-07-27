using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using System.Xml;

namespace NeuralNetwork
{
    class NeuralNet
    {
        const double _ETA = 0.15; // скорость обучения
        const double _ALPHA = 0.5; // момент
        const int _EPOCH = 3000; // эпоха

        public int HiddenLayersCount
        {
            get
            {
                int k = 0;
                for (int i = 1; i < topology.Length - 1; i++)
                {
                    k++;
                }
                return k;
            }
            private set { }
        }
        public int NeuronsCount
        {
            get
            {
                int s = 0;
                for (int i = 0; i < topology.Length; i++)
                {
                    s += topology[i];
                }
                return s;
            }
            private set { }
        }
        public bool BiasNeuronExist
        {
            get { return true; }
            private set { }
        }

        private int[] topology;

        private double[][] outputs;
        private double[][] gradients;
        private double[][][] weights;
        private double[][][] deltaWeights;

        public NeuralNet()
        {

        }

        public NeuralNet(int[] topology)
        {
            this.topology = topology;
            Random r = new Random(DateTime.Now.Millisecond);
            ArraysInitialization(r);
        }

        private void ArraysInitialization(double[][][] weights)
        {
            int bies = 0;
            outputs = new double[topology.Length][];
            gradients = new double[topology.Length][];
            this.weights = weights;
            deltaWeights = new double[topology.Length - 1][][];

            for (int layerNum = 0; layerNum < topology.Length; layerNum++)
            {
                bies = layerNum == topology.Length - 1 ? 0 : 1;

                outputs[layerNum] = new double[topology[layerNum] + bies];
                outputs[layerNum][outputs[layerNum].Length - 1] = bies;
                gradients[layerNum] = new double[topology[layerNum] + bies];

                if (layerNum < topology.Length - 1)
                {
                    deltaWeights[layerNum] = new double[topology[layerNum] + bies][];
                    for (int n = 0; n < topology[layerNum] + bies; n++)
                    {
                        deltaWeights[layerNum][n] = new double[topology[layerNum + 1]];
                    }
                }
            }
        }

        private void ArraysInitialization(Random r)
        {
            int bies = 0;
            outputs = new double[topology.Length][];
            gradients = new double[topology.Length][];
            weights = new double[topology.Length - 1][][];
            deltaWeights = new double[topology.Length - 1][][];

            for (int layerNum = 0; layerNum < topology.Length; layerNum++)
            {
                bies = layerNum == topology.Length - 1 ? 0 : 1;

                outputs[layerNum] = new double[topology[layerNum] + bies];
                outputs[layerNum][outputs[layerNum].Length - 1] = bies;
                gradients[layerNum] = new double[topology[layerNum] + bies];

                if (layerNum < topology.Length - 1)
                {
                    weights[layerNum] = new double[topology[layerNum] + bies][];
                    deltaWeights[layerNum] = new double[topology[layerNum] + bies][];
                    for (int n = 0; n < topology[layerNum] + bies; n++)
                    {
                        weights[layerNum][n] = new double[topology[layerNum + 1]];
                        deltaWeights[layerNum][n] = new double[topology[layerNum + 1]];
                        for (int w = 0; w < weights[layerNum][n].Length; w++)
                        {
                            weights[layerNum][n][w] = r.NextDouble();
                        }
                    }
                }
            }
        }


        public double[] Ask(double[] input)
        {
            if (weights == null)
            {
                throw new Exception("Сеть не существует");
            }
            FeedForward(input);
            return GetResult();
        }


        public void Training(double[][][] data)
        {
            string consolLog = "";
            double epochError = 0;
            double setError = 0;
            List<string> log = new List<string>();

            for (int epoch = 0; /*epoch < _EPOCH*/; epoch++)
            {
                for (int trainSet = 0; trainSet < data.Length; trainSet++)
                {
                    FeedForward(data[trainSet][0]);
                    setError = BackProp(data[trainSet][1]);

                    consolLog = GetLog(data[trainSet], setError);
                    log.Add(consolLog);
                    epochError += setError;
                    Console.WriteLine(consolLog);
                }
                epochError /= data.Length;
                log.Add("Epoch error: " + epochError + "\n\n");
                Console.WriteLine(log.Last());
                if (epochError < 0.008) break;
                epochError = 0;
            }
            File.WriteAllLines("log.txt", log.ToArray());
        }

        public double[][][] GetTrainData(string path)
        {
            int setNum = 0;
            string[] data = File.ReadAllLines(path);
            double[][][] trainData = new double[data.Length / 2][][];
            double[][] dataSet;
            double[] @in;
            double[] @out;
            MatchCollection input;
            MatchCollection output;
            for (int i = 0; i < data.Length; i += 2)
            {
                dataSet = new double[2][];
                input = Regex.Matches(data[i], @"\d");
                output = Regex.Matches(data[i + 1], @"\d");
                @in = new double[input.Count];
                for (int item = 0; item < input.Count; item++)
                {
                    @in[item] = Convert.ToDouble(input[item].ToString());
                }
                @out = new double[output.Count];
                for (int item = 0; item < output.Count; item++)
                {
                    @out[item] = Convert.ToDouble(output[item].ToString());
                }
                dataSet[0] = @in;
                dataSet[1] = @out;
                trainData[setNum] = dataSet;
                setNum++;
            }
            return trainData;
        }


        private void FeedForward(double[] input)
        {
            for (int n = 0; n < input.Length; n++)
            {
                outputs[0][n] = input[n];
            }

            double sum = 0;
            for (int layerNum = 1; layerNum < outputs.Length; layerNum++)
            {
                for (int n = -1; n < outputs[layerNum].Length - 1; ++n)
                {
                    if (n == -1) n++;
                    for (int prevN = 0; prevN < weights[layerNum-1].Length; prevN++)
                    {
                        sum += outputs[layerNum - 1][prevN] * weights[layerNum - 1][prevN][n];
                    }
                    outputs[layerNum][n] = ActivFunc(sum);
                    sum = 0;
                }
            }
        }

        private double BackProp(double[] targetVals)
        {
            double delta = 0;
            double error = 0;
            double[] outputLayer = outputs.Last();
            for (int n = 0; n < outputLayer.Length; n++)
            {
                delta = targetVals[n] - outputLayer[n];
                error += delta * delta;
            }
            error /= outputLayer.Length;
            error = Math.Sqrt(error);

            for (int layerNum = outputs.Length - 1; layerNum >= 0; layerNum--)
            {
                for (int n = 0; n < outputs[layerNum].Length; n++)
                {
                    if (layerNum == outputs.Length - 1)
                    {
                        gradients[layerNum][n] = (targetVals[n] - outputLayer[n]) * ActivFuncDeriv(outputs[layerNum][n]);
                    }
                    else if (layerNum != 0)
                    {
                        gradients[layerNum][n] = ActivFuncDeriv(outputs[layerNum][n]) * SumDOW(weights[layerNum][n], gradients[layerNum + 1]);
                        UpdateNeuronWeights(layerNum, n);
                    }
                    else
                    {
                        UpdateNeuronWeights(layerNum, n);
                    }
                }
            }
            return error;
        }

        private void UpdateNeuronWeights(int layerNum, int neuronNum)
        {
            double oldDeltaWeight;
            double newDeltaWeight;

            double neuronOutput = outputs[layerNum][neuronNum];
            double[] deltaWs = deltaWeights[layerNum][neuronNum];
            double[] nextLayerGradients = gradients[layerNum+1];

            for (int w = 0; w < deltaWs.Length; w++)
            {
                oldDeltaWeight = deltaWs[w];
                newDeltaWeight = _ETA * neuronOutput * nextLayerGradients[w] + _ALPHA * deltaWs[w];

                deltaWeights[layerNum][neuronNum][w] = newDeltaWeight;
                weights[layerNum][neuronNum][w] += newDeltaWeight;
            }
        }

        private double SumDOW(double[] weights, double[] nextLayerGradients)
        {
            double sum = 0;

            for (int n = 0; n < weights.Length; n++)
            {
                sum += weights[n] * nextLayerGradients[n];
            }

            return sum;
        }

        private double ActivFunc(double x)
        {
            //return 1 / (1 + Math.Exp(-x));
            return Math.Tanh(x);
        }

        private double ActivFuncDeriv(double output)
        {
            //return (1 - output) * output;
            return 1 - output * output;
        }


        public void SaveNet(string path)
        {
            if (weights == null)
            {
                throw new Exception("Сеть не существует");
            }

            XmlDocument xDoc = new XmlDocument();

            XmlElement xRoot = xDoc.CreateElement("root");
            XmlElement xTopology = xDoc.CreateElement("topology");
            XmlElement xInput = xDoc.CreateElement("input");
            XmlElement xOutput = xDoc.CreateElement("output");
            XmlElement xHidden;
            XmlElement xLayers = xDoc.CreateElement("layers");
            XmlElement xLayer = xDoc.CreateElement("layer");
            XmlElement xNeuron = xDoc.CreateElement("neuron");
            XmlElement xSynapse = xDoc.CreateElement("synapse");

            xInput.InnerText = topology[0].ToString();
            xTopology.AppendChild(xInput);
            for (int i = 1; i < topology.Length - 1; i++)
            {
                xHidden = xDoc.CreateElement("hidden");
                xHidden.InnerText = topology[i].ToString();
                xTopology.AppendChild(xHidden);
            }
            xOutput.InnerText = topology[topology.Length - 1].ToString();
            xTopology.AppendChild(xOutput);
            xRoot.AppendChild(xTopology);

            for (int layerNum = 0; layerNum < weights.Length; layerNum++)
            {
                xLayer = xDoc.CreateElement("layer");
                for (int neuron = 0; neuron < weights[layerNum].Length; neuron++)
                {
                    xNeuron = xDoc.CreateElement("neuron");
                    for (int weight = 0; weight < weights[layerNum][neuron].Length; weight++)
                    {
                        xSynapse = xDoc.CreateElement("synapse");
                        xSynapse.SetAttribute("weight", weights[layerNum][neuron][weight].ToString());
                        xNeuron.AppendChild(xSynapse);
                    }
                    xLayer.AppendChild(xNeuron);
                }
                xLayers.AppendChild(xLayer);

            }
            xRoot.AppendChild(xLayers);
            xDoc.AppendChild(xRoot);

            xDoc.Save(path);
        }

        public void OpenNet(string path)
        {
            int[] newTopology;
            List<int> NewTopologyList = new List<int>();

            double[][][] weights;

            XmlDocument xDoc = new XmlDocument();
            xDoc.Load(path);
            XmlElement xRoot = xDoc.DocumentElement;
            XmlElement xTopology = (XmlElement)xRoot.GetElementsByTagName("topology").Item(0);
            XmlElement xLayers = (XmlElement)xRoot.GetElementsByTagName("layers").Item(0);

            NewTopologyList.Add(Convert.ToInt32(xTopology.GetElementsByTagName("input").Item(0).InnerText));
            foreach (XmlElement xHidden in xRoot.GetElementsByTagName("hidden"))
            {
                NewTopologyList.Add(Convert.ToInt32(xHidden.InnerText));
            }
            NewTopologyList.Add(Convert.ToInt32(xTopology.GetElementsByTagName("output").Item(0).InnerText));
            newTopology = NewTopologyList.ToArray();
            topology = newTopology;

            int numOutputs = 0;
            weights = new double[newTopology.Length - 1][][];
            for (int layerNum = 0; layerNum < newTopology.Length - 1; layerNum++)
            {
                XmlElement layer = (XmlElement)xLayers.GetElementsByTagName("layer").Item(layerNum);

                weights[layerNum] = new double[topology[layerNum] + 1][];

                numOutputs = layerNum == newTopology.Length - 1 ? 0 : newTopology[layerNum + 1];

                for (int neuronNum = 0; neuronNum < newTopology[layerNum] + 1; neuronNum++)
                {
                    XmlElement neuron = (XmlElement)layer.GetElementsByTagName("neuron").Item(neuronNum);
                    weights[layerNum][neuronNum] = GetNeuronWeights(neuron);
                }
            }
            ArraysInitialization(weights);
        }


        private double[] GetNeuronWeights(XmlElement neuron)
        {
            List<double> weights = new List<double>();
            foreach (XmlElement synapse in neuron.ChildNodes)
            {
                weights.Add(Convert.ToDouble(synapse.GetAttribute("weight")));
            }
            return weights.ToArray();
        }

        private string GetLog(double[][] set, double setError)
        {
            double[] res;
            string input = "";
            string ideal = "";
            string output = "";
            input += "input: ";
            for (int i = 0; i < set[0].Length; i++)
            {
                input += set[0][i] + " ";
            }
            input += "\n";
            ideal += "ideal: ";
            for (int i = 0; i < set[1].Length; i++)
            {
                ideal += set[1][i] + " ";
            }
            ideal += "\n";
            res = GetResult();
            output += "output: ";
            for (int i = 0; i < res.Length; i++)
            {
                output += res[i] + " ";
            }
            output += "\nSet error: " + setError + "\n";
            return input + ideal + output;
        }

        private double[] GetResult()
        {
            return outputs.Last();
        }
    }
}
