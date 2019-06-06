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
	public static double[] sumExpValues; // Partition function values ��͵�ֵ
	public static double[][] dotProductValues;   //��˷���ֵ
	public static double[][] expDotProductValues;  //ָ���仯���ֵ
	public static double[][] topicVectors;// Vector representations for topics ����������ʾ
	public static int vocabularySize; // The number of word types in the corpus  ���������е��ʵ�����
	public static int vectorSize; // Number of vector dimensions  ������ά��
	public static HashMap<String, Integer> word2IdVocabulary; // Vocabulary to get ID  ���ʵı��
	// given a word
	public static HashMap<Integer, String> id2WordVocabulary; // Vocabulary to get word  �����ת��Ϊ����  �������
	public static void main(String[] args) throws Exception {
		System.out.println(getWordProbGenrateNewTopic());
		System.out.println(1.000000/vocabularySize);
	}
	public static double getWordProbGenrateNewTopic() throws Exception{
		//��ȡ����
		readcorpusFile("cardata/ccomment_combine.txt");
		readWordVectors("cardata/wordVectors.txt");
		//�Ż�
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
				//����������ʾ  ��������ĵ��ʸ���  ������ ����ֵ(�����Ǵ������-----�Ա�ִ��TopicVectorOptimizer)
				TopicVectorOptimizer optimizer = new TopicVectorOptimizer(
						topicVectors[0], topicWordCountLF[0], wordVectors, l2Value);
				//ͨ��LBFGS�Ż�
				Optimizer gd = new LBFGS(optimizer, tolerance);
				gd.optimize(600);
				//��Ҫ�Ż��Ĳ���
				optimizer.getParameters(topicVectors[0]);
				//����������������ĳ˻��Լ���Ӻ�-----���ÿ���������һ������ֵ��Ϊ�˸�������ʹ�ã�
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
		//��ת��Ϊ���
		word2IdVocabulary = new HashMap<String, Integer>();
		//���ת��Ϊ��
		id2WordVocabulary = new HashMap<Integer, String>();
		BufferedReader br = null;
		try {
			int indexWord = -1;
			br = new BufferedReader(new FileReader(pathTocorpusFile));
			//ÿһ�б�ʾһ���ĵ�
			for (String doc; (doc = br.readLine()) != null;) {
				if (doc.trim().length() == 0)
					continue;
				//�ĵ����ʲ��
				String[] words = doc.trim().split("\\s+");
				//�ĵ���ʾ�ɼ���
				List<Integer> document = new ArrayList<Integer>();
				//���ĵ������е��ʽ���ѭ��
				for (String word : words) {
					//�ĵ��е��ʱ��-----�����ȫ�ֶ���
					if (word2IdVocabulary.containsKey(word)) {
						//��������˸õ��ʣ����õ���ֱ����ӵ��ĵ�������
						document.add(word2IdVocabulary.get(word));
					}
					else {
						//��1��ʾ��0��ʼ�Ե��ʽ��б�ţ�������Ŷ�Ӧ�ĵ��ʼ��뵽id2WordVocabulary
						indexWord += 1;
						word2IdVocabulary.put(word, indexWord);
						id2WordVocabulary.put(indexWord, word);
						//�ĵ���Ӹõ���
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
	//��ȡ�������ļ�
	public static void readWordVectors(String pathToWordVectorsFile)
			throws Exception
	{
		//�����Ҫ��ȡ�������ļ�����Ե�ַ
		System.out.println("Reading word vectors from word-vectors file " + pathToWordVectorsFile
				+ "...");

		BufferedReader br = null;
		try {
			br = new BufferedReader(new FileReader(pathToWordVectorsFile));
			//�Կո�ֿ�
			String[] elements = br.readLine().trim().split("\\s+");
			//�������ĳ��ȣ������1����Ϊ��һά���Ǵ�
			vectorSize = elements.length - 1;
			//word2vec������ά�ȣ�ֻȥ�������еĴ�vocabularySize
			wordVectors = new double[vocabularySize][vectorSize];
			//����Ϊ��һά��
			String word = elements[0];
			//�����������������еĻ������ôʵĴ�������������wordVectors
			if (word2IdVocabulary.containsKey(word)) {
				for (int j = 0; j < vectorSize; j++) {
					wordVectors[word2IdVocabulary.get(word)][j] = new Double(elements[j + 1]);
				}
			}
			//�������ı�������֮����Ҫ�ȶ�һ����Ϊ�˳�ʼ������ȡ��������ά��
			for (String line; (line = br.readLine()) != null;) {
				elements = line.trim().split("\\s+");
				word = elements[0];
				//�����г��ֵ�ÿ�����ʵĴ�����
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
		//��ֹ�����еĴ���word2vec�ļ��в�����
		for (int i = 0; i < vocabularySize; i++) {
			if (MatrixOps.absNorm(wordVectors[i]) == 0.0) {
				System.out.println("The word \"" + id2WordVocabulary.get(i)
				+ "\" doesn't have a corresponding vector!!!");
				throw new Exception();
			}
		}
	}
}
