package dplftm;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import cc.mallet.optimize.InvalidOptimizableException;
import cc.mallet.optimize.Optimizer;
import cc.mallet.types.MatrixOps;
import utility.LBFGS;

public class NewTopicVectorsoptimize {
	public final static double l2Regularizer = 0.01;
	public final static double tolerance = 0.05;
	public static int[][] topicWordCountLF;
	public static double[][] wordVectors;
	public static double[] sumExpValues; // Partition function values 求和的值
	public static double[][] dotProductValues;   //点乘法的值
	public static double[][] expDotProductValues;  //指数变化后的值
	public static double[][] topicVectors;// Vector representations for topics 主题向量表示
	public static int vocabularySize; // The number of word types in the corpus  整个语料中单词的总数
	public static int vectorSize; // Number of vector dimensions  向量的维度
	public static HashMap<String, Integer> word2IdVocabulary; // Vocabulary to get ID  单词的编号
	// given a word
	public static HashMap<Integer, String> id2WordVocabulary; // Vocabulary to get word  将编号转化为单词  用于输出
	public static void main(String[] args) throws Exception {
		System.out.println(getWordProbGenrateNewTopic());
		System.out.println(1.000000/vocabularySize);
	}
	public static double getWordProbGenrateNewTopic() throws Exception{
		//读取语料
		readcorpusFile("cardata/ccomment_combine.txt");
		readWordVectors("cardata/wordVectors.txt");
		//优化
		topicWordCountLF = new int[1][vocabularySize];
		for (int i = 0; i < vocabularySize; i++) {
			topicWordCountLF[0][i] = 0;
		}
		optimizeTopicVectors();
		return expDotProductValues[0][0]/ sumExpValues[0];
		
	}
	public static void optimizeTopicVectors()
	{
		System.out.println("\t\tEstimating new topic vectors ...");
		sumExpValues = new double[1];
		dotProductValues = new double[1][vocabularySize];
		expDotProductValues = new double[1][vocabularySize];
		topicVectors = new double[1][vectorSize];

		int rate = 1;
		boolean check = true;
		while (check) {
			double l2Value = l2Regularizer * rate;
			try {
				//主题向量表示  主题包含的单词个数  词向量 正则化值(这里是传入参数-----以便执行TopicVectorOptimizer)
				TopicVectorOptimizer optimizer = new TopicVectorOptimizer(
						topicVectors[0], topicWordCountLF[0], wordVectors, l2Value);
				//通过LBFGS优化
				Optimizer gd = new LBFGS(optimizer, tolerance);
				gd.optimize(600);
				//需要优化的参数
				optimizer.getParameters(topicVectors[0]);
				//输入的是两个特征的乘积以及其加和-----针对每个主题计算一个向量值（为了更新主题使用）
				sumExpValues[0] = optimizer.computePartitionFunction(
						dotProductValues[0], expDotProductValues[0]);
				check = false;

				if (sumExpValues[0] == 0 || Double.isInfinite(sumExpValues[0])) {
					double max = -1000000000.0;
					for (int index = 0; index < vocabularySize; index++) {
						if (dotProductValues[0][index] > max)
							max = dotProductValues[0][index];
					}
					for (int index = 0; index < vocabularySize; index++) {
						expDotProductValues[0][index] = Math
								.exp(dotProductValues[0][index] - max);
						sumExpValues[0] += expDotProductValues[0][index];
					}
				}
			}
			catch (InvalidOptimizableException e) {
				e.printStackTrace();
				check = true;
			}
			rate = rate * 10;
		}
	}
	public static void readcorpusFile(String pathTocorpusFile)
	{
		//词转化为编号
		word2IdVocabulary = new HashMap<String, Integer>();
		//编号转化为词
		id2WordVocabulary = new HashMap<Integer, String>();
		BufferedReader br = null;
		try {
			int indexWord = -1;
			br = new BufferedReader(new FileReader(pathTocorpusFile));
			//每一行表示一个文档
			for (String doc; (doc = br.readLine()) != null;) {
				if (doc.trim().length() == 0)
					continue;
				//文档单词拆分
				String[] words = doc.trim().split("\\s+");
				//文档表示成集合
				List<Integer> document = new ArrayList<Integer>();
				//对文档的所有单词进行循环
				for (String word : words) {
					//文档中单词编号-----编号是全局而言
					if (word2IdVocabulary.containsKey(word)) {
						//如果包含了该单词，将该单词直接添加到文档集合中
						document.add(word2IdVocabulary.get(word));
					}
					else {
						//加1表示从0开始对单词进行编号，并将编号对应的单词加入到id2WordVocabulary
						indexWord += 1;
						word2IdVocabulary.put(word, indexWord);
						id2WordVocabulary.put(indexWord, word);
						//文档添加该单词
						document.add(indexWord);
					}
				}
//				corpus.add(document);
			}
		}
		catch (Exception e) {
			e.printStackTrace();
		}
		vocabularySize = word2IdVocabulary.size();
	}
	//读取词向量文件
	public static void readWordVectors(String pathToWordVectorsFile)
			throws Exception
	{
		//输出需要读取词向量文件的相对地址
		System.out.println("Reading word vectors from word-vectors file " + pathToWordVectorsFile
				+ "...");

		BufferedReader br = null;
		try {
			br = new BufferedReader(new FileReader(pathToWordVectorsFile));
			//以空格分开
			String[] elements = br.readLine().trim().split("\\s+");
			//词向量的长度，这里减1是因为第一维度是词
			vectorSize = elements.length - 1;
			//word2vec向量的维度，只去语料中有的词vocabularySize
			wordVectors = new double[vocabularySize][vectorSize];
			//单词为第一维度
			String word = elements[0];
			//如果这个词语在语料中的话，将该词的词向量存入数组wordVectors
			if (word2IdVocabulary.containsKey(word)) {
				for (int j = 0; j < vectorSize; j++) {
					wordVectors[word2IdVocabulary.get(word)][j] = new Double(elements[j + 1]);
				}
			}
			//继续读文本，上面之所以要先读一行是为了初始化，获取词向量的维度
			for (String line; (line = br.readLine()) != null;) {
				elements = line.trim().split("\\s+");
				word = elements[0];
				//语料中出现的每个单词的词向量
				if (word2IdVocabulary.containsKey(word)) {
					for (int j = 0; j < vectorSize; j++) {
						wordVectors[word2IdVocabulary.get(word)][j] = new Double(elements[j + 1]);
					}
				}
			}
		}
		catch (Exception e) {
			e.printStackTrace();
		}
		//防止语料中的词在word2vec文件中不存在
		for (int i = 0; i < vocabularySize; i++) {
			if (MatrixOps.absNorm(wordVectors[i]) == 0.0) {
				System.out.println("The word \"" + id2WordVocabulary.get(i)
				+ "\" doesn't have a corresponding vector!!!");
				throw new Exception();
			}
		}
	}
}
