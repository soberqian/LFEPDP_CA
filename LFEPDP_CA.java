package dplftm;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

import utility.FuncUtils;
import utility.LBFGS;
import utility.Parallel;
import cc.mallet.optimize.InvalidOptimizableException;
import cc.mallet.optimize.Optimizer;
import cc.mallet.types.MatrixOps;
import cc.mallet.util.Randoms;

/**
 * We proposed the LF-EPDP model for mining product relationships.
 * 
 * This is the implementation of this model.
 * 
 * We also reference the code of the LFDMM.
 * 
 * @author Yang Qian
 */

public class LFEPDP_CA
{
	public List<String> iter_topic;
	public double alpha; // Hyper-parameter alpha  ������
	public double beta; // Hyper-parameter beta   ������
	// public double alphaSum; // alpha * numTopics
	public double betaSum; // beta * vocabularySize  V*beta

	public int numTopics; // Number of topics  ������Ŀ
	public int topWords; // Number of most probable words for each topic  ÿ������ȡ���ٸ���ǰ�ĵ���

	public double lambda; // Mixture weight value  ���Ȩ��ֵ
	public int numInitIterations;  //
	public int numIterations; // Number of EM-style sampling iterations  ��������

	public List<List<Integer>> corpus; // Word ID-based corpus  ���ϵ��ʵ�id
	public int[] z; //�ĵ���Ӧ������
	public int[][] z_words; //�ĵ���ÿ�����ʶ�Ӧ�������⣬�ڶ�ά����0-1��ʾ��0��ʾ�����ڶ���ʽ�ֲ���1��ʾ������������
	// in the corpus
	public int numDocuments; // Number of documents in the corpus  �ĵ�������
	public int numWordsInCorpus; // Number of words in the corpus  �������ϵ��ʵ�����

	public HashMap<String, Integer> word2IdVocabulary; // Vocabulary to get ID  ���ʵı��
	// given a word
	public HashMap<Integer, String> id2WordVocabulary; // Vocabulary to get word  �����ת��Ϊ����  �������
	// given an ID
	public int vocabularySize; // The number of word types in the corpus  ���������е��ʵ�����

	// Number of documents assigned to a topic  ���䵽һ�������ĵ�������
	public int[] docTopicCount;
	// numTopics * vocabularySize matrix
	// Given a topic: number of times a word type generated from the topic by
	// the Dirichlet multinomial component  �����Ӧ�ĵ�������  �õ������ɶ���ʽ�ֲ�����
	public int[][] topicWordCountDMM;
	// Total number of words generated from each topic by the Dirichlet
	// multinomial component �����Ӧ���ܵĵ�������  ��Щ�������ɶ���ʽ�ֲ�����
	public int[] sumTopicWordCountDMM;
	// numTopics * vocabularySize matrix
	// Given a topic: number of times a word type generated from the topic by
	// the latent feature component  �����������������ֲ���  ͳ��һ�������Ӧ�ĵ�������
	public int[][] topicWordCountLF;
	// Total number of words generated from each topic by the latent feature
	// component  ����������������  ͳ��һ�������Ӧ���ܵ�������
	public int[] sumTopicWordCountLF;
	// Double array used to sample a topic   ���� ���ڳ���
	public double[] multiPros;
	// Path to the directory containing the corpus  
	public String folderPath;
	// Path to the topic modeling corpus
	public String corpusPath;
	public String vectorFilePath;
	public double[][] wordVectors; // Vector representations for words  ��������ʾ
	public double[][] topicVectors;// Vector representations for topics ����������ʾ
	public int vectorSize; // Number of vector dimensions  ������ά��
	public double[][] dotProductValues;   //��˷���ֵ
	public double[][] expDotProductValues;  //ָ���仯���ֵ
	public double[] sumExpValues; // Partition function values ��͵�ֵ

	public final double l2Regularizer = 0.01; // L2 regularizer value for learning topic vectors L2����
	public final double tolerance = 0.05; // Tolerance value for LBFGS convergence  LBFGS����
	public String expName = "LFDMM";
	public String orgExpName = "LFDMM";
	public String tAssignsFilePath = "";
	public int savestep = 0;
	public LFEPDP_CA(String pathToCorpus, String pathToWordVectorsFile, int inNumTopics,
			double inAlpha, double inBeta, double inLambda, int inNumInitIterations,
			int inNumIterations, int inTopWords)
					throws Exception
	{
		this(pathToCorpus, pathToWordVectorsFile, inNumTopics, inAlpha, inBeta, inLambda,
				inNumInitIterations, inNumIterations, inTopWords, "LFDMM");
	}

	public LFEPDP_CA(String pathToCorpus, String pathToWordVectorsFile, int inNumTopics,
			double inAlpha, double inBeta, double inLambda, int inNumInitIterations,
			int inNumIterations, int inTopWords, String inExpName)
					throws Exception
	{
		this(pathToCorpus, pathToWordVectorsFile, inNumTopics, inAlpha, inBeta, inLambda,
				inNumInitIterations, inNumIterations, inTopWords, inExpName, "", 0);
	}

	public LFEPDP_CA(String pathToCorpus, String pathToWordVectorsFile, int inNumTopics,
			double inAlpha, double inBeta, double inLambda, int inNumInitIterations,
			int inNumIterations, int inTopWords, String inExpName, String pathToTAfile)
					throws Exception
	{
		this(pathToCorpus, pathToWordVectorsFile, inNumTopics, inAlpha, inBeta, inLambda,
				inNumInitIterations, inNumIterations, inTopWords, inExpName, pathToTAfile, 0);
	}

	public LFEPDP_CA(String pathToCorpus, String pathToWordVectorsFile, int inNumTopics,
			double inAlpha, double inBeta, double inLambda, int inNumInitIterations,
			int inNumIterations, int inTopWords, String inExpName, int inSaveStep)
					throws Exception
	{
		this(pathToCorpus, pathToWordVectorsFile, inNumTopics, inAlpha, inBeta, inLambda,
				inNumInitIterations, inNumIterations, inTopWords, inExpName, "", inSaveStep);
	}
	//�����ʼ��
	public LFEPDP_CA(String pathToCorpus, String pathToWordVectorsFile, int inNumTopics,
			double inAlpha, double inBeta, double inLambda, int inNumInitIterations,
			int inNumIterations, int inTopWords, String inExpName, String pathToTAfile,
			int inSaveStep)
					throws Exception
	{
		iter_topic = new ArrayList<String>();
		alpha = inAlpha;
		beta = inBeta;
		lambda = inLambda;
		numTopics = inNumTopics;
		numIterations = inNumIterations;
		numInitIterations = inNumInitIterations;
		topWords = inTopWords;
		savestep = inSaveStep;
		expName = inExpName;
		orgExpName = expName;
		//word2vec����
		vectorFilePath = pathToWordVectorsFile;
		//���ϵ�·��
		corpusPath = pathToCorpus;
		folderPath = pathToCorpus.substring(0,
				Math.max(pathToCorpus.lastIndexOf("/"), pathToCorpus.lastIndexOf("\\")) + 1);
		//�������ϵ�·��
		System.out.println("Reading topic modeling corpus: " + pathToCorpus);
		//��ת��Ϊ���
		word2IdVocabulary = new HashMap<String, Integer>();
		//���ת��Ϊ��
		id2WordVocabulary = new HashMap<Integer, String>();
		//����
		corpus = new ArrayList<List<Integer>>();
		//�ĵ���Ŀ
		numDocuments = 0;
		//�����е��ʵ���Ŀ
		numWordsInCorpus = 0;
		//��ȡ����
		BufferedReader br = null;
		try {
			int indexWord = -1;
			br = new BufferedReader(new FileReader(pathToCorpus));
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
				//�ĵ���Ŀ++
				numDocuments++;
				//���������е��ʵ�����
				numWordsInCorpus += document.size();
				//�������ĵ���ӵ�������
				corpus.add(document);
			}
		}
		catch (Exception e) {
			e.printStackTrace();
		}
		//���ϲ��ظ����ʵ�����
		vocabularySize = word2IdVocabulary.size();
		//�����Ӧ���ĵ�ͳ��
		docTopicCount = new int[numTopics];
		//����-����ͳ��  ���Զ���ʽ�ֲ�
		topicWordCountDMM = new int[numTopics][vocabularySize];
		//�����Ӧ�ĵ�������Ŀͳ�� ���Զ���ʽ�ֲ�
		sumTopicWordCountDMM = new int[numTopics];
		//����-����ͳ�� �������������ֲ�
		topicWordCountLF = new int[numTopics][vocabularySize];
		//�����Ӧ�ĵ�����Ŀ�ܼ� ������������
		sumTopicWordCountLF = new int[numTopics];
		//����ʽ�ֲ�������
		multiPros = new double[numTopics];
		//����Ϊ1/K,����Ҫ���̶ĵģ��ڳ�ʼ����ʱ��
		for (int i = 0; i < numTopics; i++) {
			multiPros[i] = 1.0 / numTopics;
		}

		// alphaSum = numTopics * alpha;  
		betaSum = vocabularySize * beta;  
		//��ȡ������ word2vec�ļ�
		readWordVectorsFile(vectorFilePath);
		topicVectors = new double[numTopics][vectorSize];
		dotProductValues = new double[numTopics][vocabularySize];
		expDotProductValues = new double[numTopics][vocabularySize];
		sumExpValues = new double[numTopics];

		System.out
		.println("Corpus size: " + numDocuments + " docs, " + numWordsInCorpus + " words");
		System.out.println("Vocabuary size: " + vocabularySize);
		System.out.println("Number of topics: " + numTopics);
		System.out.println("alpha: " + alpha);
		System.out.println("beta: " + beta);
		System.out.println("lambda: " + lambda);
		System.out.println("Number of initial sampling iterations: " + numInitIterations);
		System.out.println("Number of EM-style sampling iterations for the LF-DMM model: "
				+ numIterations);
		System.out.println("Number of top topical words: " + topWords);

		tAssignsFilePath = pathToTAfile;
		initialize();

	}
	//��ȡ�������ļ�
	public void readWordVectorsFile(String pathToWordVectorsFile)
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
	//��ʼ������
	public void initialize()
			throws IOException
	{
		//������ĵ������������
		System.out.println("Randomly initialzing topic assignments ...");
		z = new  int[numDocuments];
		z_words = new  int[numDocuments][];  //��ʼ��
		//ѭ��ÿƪ�ĵ�
		for (int docId = 0; docId < numDocuments; docId++) {
			//�������̶Ļ�ȡ�����ţ�ǰ���Ѿ���ʼ���ˣ�������multiPros������ֵ��������ᱨ��
			int topic= FuncUtils.nextDiscrete(multiPros);
			z[docId] = topic; //��ֵ����
			//���䵽��������ĵ�����+1
			docTopicCount[topic] += 1;
			//�ĵ��ĵ��ʸ���
			int docSize = corpus.get(docId).size();
			z_words[docId] = new int[docSize];  //��ÿƪ�ĵ�������һ�γ�ʼ�������Ҫϰ��ʹ����
			//ѭ��ÿ������
			for (int j = 0; j < docSize; j++) {
				//��ȡ���ʱ��
				int wordId = corpus.get(docId).get(j);
				//�������false or true,������ʼ�����ĵ������������������Ƕ���ʽ�ֲ�
				boolean component = new Randoms().nextBoolean();
				if (!component) { // Generated from the latent feature component
					//����-������������1  ����������������
					topicWordCountLF[topic][wordId] += 1;
					// ���������ɵĵ�����������1 ����������������
					sumTopicWordCountLF[topic] += 1;
					z_words[docId][j] = 1;//1��ʾ�������������ֲ�
				}
				else {// Generated from the Dirichlet multinomial component
					//����-������������1 �ɶ���ʽ�ֲ�����
					topicWordCountDMM[topic][wordId] += 1;
					//�������ɵĵ�����������1
					sumTopicWordCountDMM[topic] += 1;
					z_words[docId][j] = 0; //��ʾ�����ڶ���ʽ�ֲ�
				}
			}
		}
	}
	//ģ���ƶ�
	public void inference()
			throws IOException
	{
		System.out.println("Running Gibbs sampling inference: ");
		//��ʼ������
		for (int iter = 1; iter <= numInitIterations; iter++) {

			System.out.println("\tInitial sampling iteration: " + (iter));
			//���ʳ�ʼ������
			sampleSingleInitialIteration();
		}
		optimizeTopicVectors();
		//��ʽ����
		for (int iter = 1; iter <= numIterations; iter++) {
			//			intialize_cluster(docTopicCount,topicWordCountDMM,topicWordCountLF,sumTopicWordCountDMM,sumTopicWordCountLF);
			iter_topic.add(iter + "\t" + numTopics);
			System.out.println("\tLFDMM sampling iteration: " + (iter)+"\tnumber of topic:"+numTopics);
			optimizeTopicVectors();
			//��ȡ����
			sampleSingleIteration();
			defragment();
			if ((savestep > 0) && (iter % savestep == 0) && (iter < numIterations)) {
				System.out.println("\t\tSaving the output from the " + iter + "^{th} sample");
				expName = orgExpName + "-" + iter;
				write();
			}
		}
		expName = orgExpName;
		//����ģ����ز���
		writeParameters();
		System.out.println("Writing output from the last sample ...");
		//������Ϣ
		write();

		System.out.println("Sampling completed!");
	}
	//�Ż���������
	public void optimizeTopicVectors()
	{
		System.out.println("\t\tEstimating topic vectors ...");
		sumExpValues = new double[numTopics];
		dotProductValues = new double[numTopics][vocabularySize];
		expDotProductValues = new double[numTopics][vocabularySize];

		Parallel.loop(numTopics, new Parallel.LoopInt()
		{
			@Override
			public void compute(int topic)
			{
				int rate = 1;
				boolean check = true;
				while (check) {
					double l2Value = l2Regularizer * rate;
					try {
						//����������ʾ  ��������ĵ��ʸ���  ������ ����ֵ(�����Ǵ������-----�Ա�ִ��TopicVectorOptimizer)
						TopicVectorOptimizer optimizer = new TopicVectorOptimizer(
								topicVectors[topic], topicWordCountLF[topic], wordVectors, l2Value);
						//ͨ��LBFGS�Ż�
						Optimizer gd = new LBFGS(optimizer, tolerance);
						gd.optimize(600);
						//��Ҫ�Ż��Ĳ���
						optimizer.getParameters(topicVectors[topic]);
						//����������������ĳ˻��Լ���Ӻ�-----���ÿ���������һ������ֵ��Ϊ�˸�������ʹ�ã�
						sumExpValues[topic] = optimizer.computePartitionFunction(
								dotProductValues[topic], expDotProductValues[topic]);
						check = false;

						if (sumExpValues[topic] == 0 || Double.isInfinite(sumExpValues[topic])) {
							double max = -1000000000.0;
							for (int index = 0; index < vocabularySize; index++) {
								if (dotProductValues[topic][index] > max)
									max = dotProductValues[topic][index];
							}
							for (int index = 0; index < vocabularySize; index++) {
								expDotProductValues[topic][index] = Math
										.exp(dotProductValues[topic][index] - max);
								sumExpValues[topic] += expDotProductValues[topic][index];
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
		});
	}
	//�Ż�ĳһ�����������
	public  void optimizeTopicVectorsKK(int k)
	{
		System.out.println("\t\tEstimating new topic vectors ...");
		int rate = 1;
		boolean check = true;
		while (check) {
			double l2Value = l2Regularizer * rate;
			try {
				//����������ʾ  ��������ĵ��ʸ���  ������ ����ֵ(�����Ǵ������-----�Ա�ִ��TopicVectorOptimizer)
				TopicVectorOptimizer optimizer = new TopicVectorOptimizer(
						topicVectors[k], topicWordCountLF[k], wordVectors, l2Value);
				//ͨ��LBFGS�Ż�
				Optimizer gd = new LBFGS(optimizer, tolerance);
				gd.optimize(600);
				//��Ҫ�Ż��Ĳ���
				optimizer.getParameters(topicVectors[k]);
				//����������������ĳ˻��Լ���Ӻ�-----���ÿ���������һ������ֵ��Ϊ�˸�������ʹ�ã�
				sumExpValues[k] = optimizer.computePartitionFunction(
						dotProductValues[k], expDotProductValues[k]);
				check = false;

				if (sumExpValues[k] == 0 || Double.isInfinite(sumExpValues[k])) {
					double max = -1000000000.0;
					for (int index = 0; index < vocabularySize; index++) {
						if (dotProductValues[k][index] > max)
							max = dotProductValues[k][index];
					}
					for (int index = 0; index < vocabularySize; index++) {
						expDotProductValues[k][index] = Math
								.exp(dotProductValues[k][index] - max);
						sumExpValues[k] += expDotProductValues[k][index];
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
	/**
	 * @param �������Ϊ(1)�����Ӧ���ĵ�����ͳ��;(2)����������ʵ�ͳ�ƣ������������������������Ͷ���ʽ�ֲ����ɣ�3������������ܵ���ͳ�ƣ���������ʽ
	 * �ֲ����ɺ��������ֲ�����
	 * @return
	 * @Date: 2018-3-29
	 * @Author: yang qian 
	 * @Description: remove a topic
	 */
	//�Ƴ�����,�������±�ţ��������·���------���������������Ż�����Ҳ���������
	public void intialize_cluster(int[] docTopicCount,int[][] topicWordCountDMM,int[][] topicWordCountLF, 
			int[] sumTopicWordCountDMM, int[] sumTopicWordCountLF){
		//�������·���
		HashMap<Integer, Integer> countk = new HashMap<Integer, Integer>(); 
		int j=-1;
		for(int i=0;i<numDocuments;i++){
			if (countk.containsKey(z[i])){
				countk.put(z[i],countk.get(z[i]));
			}else{  
				j++;
				countk.put(z[i], j);
			}
		}
		//������Ŀ�ᷢ���仯
		numTopics=countk.keySet().size();
		//		System.out.println(numTopics);
		//�����Ӧ���ĵ�����
		for(int i=0;i<numDocuments;i++){
			z[i]=countk.get(z[i]); //Map��valueֵ
		}
		this.docTopicCount = new int [numTopics]; //�����Ӧ���ĵ�����
		this.topicWordCountDMM = new int [numTopics][vocabularySize]; //����z��Ӧ�ĵ���v�������������ڶ���ʽ�ֲ�
		this.topicWordCountLF = new int [numTopics][vocabularySize];; //����z��Ӧ�ĵ���v�������������ڶ���ʽ�ֲ�
		this.sumTopicWordCountDMM = new int[numTopics];
		this.sumTopicWordCountLF = new int[numTopics];
		//ѭ��ÿ�����⣬����س�ʼ����ֵ
		for(int k = 0; k < numTopics; k++){
			this.docTopicCount[k] = 0;
			this.sumTopicWordCountDMM[k] = 0;
			this.sumTopicWordCountLF[k] = 0;
			for(int t = 0; t < vocabularySize; t++){
				this.topicWordCountDMM[k][t] = 0;
				this.topicWordCountLF[k][t] = 0;
			}
		}
		//����ͳ��ֵ
		for(int dIndex = 0; dIndex < numDocuments; dIndex++){
			List<Integer> document = corpus.get(dIndex); 
			int docSize = document.size();  //��ȡ�ĵ��ĵ�������
			int topic = z[dIndex];
			this.docTopicCount[topic] ++ ;
			for(int wIndex = 0; wIndex < docSize; wIndex++){
				int word = document.get(wIndex);// wordId
				int subtopic =z_words[dIndex][wIndex] ;
				if (subtopic == 1) {
					this.topicWordCountLF[topic][word] += 1;
					this.sumTopicWordCountLF[topic] += 1;
				}
				else {
					this.topicWordCountDMM[topic][word] += 1;
					this.sumTopicWordCountDMM[topic] += 1;
				}
			}
		}
		//���³�ʼ��,��������
		this.topicVectors = new double[numTopics][vectorSize];
	}
	/**
	 * @param �������в������ĵ��������Ƴ�
	 * @return
	 * @Date: 2018-3-29
	 * @Author: yang qian 
	 * @Description: remove a topic
	 */
	public void removeDocument(int docID){
		//��ȡ�ĵ������е���
		List<Integer> document = corpus.get(docID);
		//�ĵ��ĳ���
		int docSize = document.size();
		//�ĵ��ĳ�ʼ����ֲ������������Ƴ��õ���
		int topic = z[docID];
		//�����Ӧ���ĵ���Ŀ��1
		docTopicCount[topic] = docTopicCount[topic] - 1;
		for (int wIndex = 0; wIndex < docSize; wIndex++) {
			//��ȡ���ʵ�id
			int word = document.get(wIndex);// wordId
			int subtopic =z_words[docID][wIndex] ;
			if (subtopic == 1) {
				topicWordCountLF[topic][word] -= 1;
				sumTopicWordCountLF[topic] -= 1;
			}
			else {
				topicWordCountDMM[topic][word] -= 1;
				sumTopicWordCountDMM[topic] -= 1;
			}
		}
	}
	public void defragment() {
		int[] kOldToKNew = new int[numTopics];
		int k, newK = 0;
		for (k = 0; k < numTopics; k++) {
			if (docTopicCount[k] > 0) {
				kOldToKNew[k] = newK;  //
				swap(sumTopicWordCountDMM, newK, k);
				swap(sumTopicWordCountLF, newK, k);
				swap(topicWordCountDMM, newK, k);
				swap(topicWordCountLF, newK, k);
				swap(topicVectors, newK, k);
				swap(dotProductValues, newK, k);
				swap(expDotProductValues, newK, k);
				swap(sumExpValues, newK, k);
				swap(docTopicCount, newK, k);
				newK++;
			} else {

			}
		}
		numTopics = newK;
		//���ﻹҪ���´����ĵ���Ӧ������
		for (int dIndex = 0; dIndex < numDocuments; dIndex++) {
			z[dIndex] = kOldToKNew[z[dIndex]];
		}
	}
	public void swap(int[] arr, int arg1, int arg2){
		int t = arr[arg1]; 
		arr[arg1] = arr[arg2]; 
		arr[arg2] = t; 
	}
	public  void swap(int[][] arr, int arg1, int arg2) {
		int[] t = arr[arg1]; 
		arr[arg1] = arr[arg2]; 
		arr[arg2] = t; 
	}
	public  void swap(double[] arr, int arg1, int arg2){
		double t = arr[arg1]; 
		arr[arg1] = arr[arg2]; 
		arr[arg2] = t; 
	}
	public  void swap(double[][] arr, int arg1, int arg2) {
		double[] t = arr[arg1]; 
		arr[arg1] = arr[arg2]; 
		arr[arg2] = t; 
	}
	//������Ŵ�ȷ����Խ��
	public  int[] ensureCapacity(int[] arr,int i) {
		int length = arr.length;
		int[] arr2 = new int[length+i];
		System.arraycopy(arr, 0, arr2, 0, length);
		return arr2;
	}
	public  int[][] ensureCapacity(int[][] array,int i,int j) {  
		int[][] arr = new int[array.length +i][array[0].length +j];       //��չ  
		for(int c = 0; c< array.length; c++) {  
			System.arraycopy(array[c], 0, arr[c], 0, array[c].length);  //���鿽��  
		}  
		return arr;  
	} 
	public  double[] ensureCapacity(double[] arr,int i) {
		int length = arr.length;
		double[] arr2 = new double[length+i];
		System.arraycopy(arr, 0, arr2, 0, length);
		return arr2;
	}
	public  double[][] ensureCapacity(double[][] array,int i,int j) {  
		double[][] arr = new double[array.length +i][array[0].length +j];       //��չ  
		for(int c = 0; c< array.length; c++) {  
			System.arraycopy(array[c], 0, arr[c], 0, array[c].length);  //���鿽��  
		}  
		return arr;  
	} 
	//ÿһ����������
	public void sampleSingleIteration()
	{
		//��ÿһƪ�ĵ�����ѭ��
		for (int dIndex = 0; dIndex < numDocuments; dIndex++) {
			//			System.out.println("��ǰִ�е��ĵ��ǣ�"+dIndex);
			double[] prob = new double[numTopics+1];
			//��ȡ�ĵ������е���
			List<Integer> document = corpus.get(dIndex);
			//�ĵ��ĳ���
			int docSize = document.size();
			//�ĵ��ĳ�ʼ����ֲ������������Ƴ��õ���
			int topic = z[dIndex];
			//�����Ӧ���ĵ���Ŀ��1
			docTopicCount[topic] = docTopicCount[topic] - 1;
			if (docTopicCount[topic] < 0) {
				System.out.println("docTopicCount < 0 "+topic +" " +docTopicCount[topic]);
				defragment();
			}
			//�������ԶԸ����ʽ���ѭ��������ص��ʵ�ͳ�ƹ���
			for (int wIndex = 0; wIndex < docSize; wIndex++) {
				//��ȡ���ʵ�id
				int word = document.get(wIndex);// wordId
				int subtopic =z_words[dIndex][wIndex] ;
				if (subtopic == 1) {
					topicWordCountLF[topic][word] -= 1;
					sumTopicWordCountLF[topic] -= 1;
				}
				else {
					topicWordCountDMM[topic][word] -= 1;
					sumTopicWordCountDMM[topic] -= 1;
				}
			}
			// ���ĵ����ʵ�������г���,������ʣ�������Ҫ��������������ĸ�����ô����
			for (int tIndex = 0; tIndex < numTopics; tIndex++) {
				prob[tIndex] = docTopicCount[tIndex]/(numDocuments - 1 + alpha);
				for (int wIndex = 0; wIndex < docSize; wIndex++) {
					int word = document.get(wIndex);
					//���ݹ�ʽ���м��㣬�������Ĺ�ʽ������  N_{d,w}+K_{d,w}�Ĵη������⣬����Ĺ�ʽӦ����������
					prob[tIndex] *= (lambda * expDotProductValues[tIndex][word]
							/ sumExpValues[tIndex] + (1 - lambda)
							* (topicWordCountDMM[tIndex][word] + beta)
							/ (sumTopicWordCountDMM[tIndex] + betaSum));
				}
			}

			prob[numTopics]= (alpha) / (numDocuments - 1 + alpha);
			double valueOfRule = 1.0;
			//��ÿ�����ʽ���ѭ��
			for(int wIndex = 0; wIndex < docSize; wIndex++){
				valueOfRule *= 1.0/vocabularySize;  //�����������ᷢ���������Ͷ���ʽ�ֲ����ɵĸ�����һ����
			}
			prob[numTopics] = prob[numTopics] * valueOfRule ;
			//�������̶�ѡ�������еĴػ��ǾɵĴ�
			topic = FuncUtils.nextDiscrete(prob);  //���̶�û������Ϊʲô�����Խ��������
			z[dIndex] = topic;  //�ĵ���Ӧ��������
			//�жϸ������Ƿ�Ϊ������
			if(topic < numTopics){
				docTopicCount[topic] += 1;
				//��ʼ�����ͳ��
				for (int wIndex = 0; wIndex < docSize; wIndex++) {
					int word = document.get(wIndex);
					//�����Ƕ�s_{di}�ĳ��������õ���ֱ�Ӽ��㣬��û��ʹ�����̶�
					if (lambda * expDotProductValues[topic][word] / sumExpValues[topic] > (1 - lambda)
							* (topicWordCountDMM[topic][word] + beta)
							/ (sumTopicWordCountDMM[topic] + betaSum)) {
						//���������������ͳ��
						topicWordCountLF[topic][word] += 1;
						sumTopicWordCountLF[topic] += 1;
						z_words[dIndex][wIndex]=1;
					}
					else {
						//���Զ���ʽ�ֲ������ͳ��
						topicWordCountDMM[topic][word] += 1;
						sumTopicWordCountDMM[topic] += 1;
						z_words[dIndex][wIndex]=0;
					}
				}
			}else if (topic == numTopics) {   // a new topic is created
				numTopics++;  //������������
				docTopicCount= ensureCapacity(docTopicCount,1);
				topicWordCountDMM = ensureCapacity(topicWordCountDMM,1,0);
				sumTopicWordCountDMM = ensureCapacity(sumTopicWordCountDMM,1);
				topicWordCountLF = ensureCapacity(topicWordCountLF,1,0);
				sumTopicWordCountLF = ensureCapacity(sumTopicWordCountLF,1);
				topicVectors = ensureCapacity(topicVectors,1,0);
				dotProductValues = ensureCapacity(dotProductValues,1,0);
				expDotProductValues = ensureCapacity(expDotProductValues,1,0);
				sumExpValues = ensureCapacity(sumExpValues,1);
				docTopicCount[numTopics-1] += 1;
				//�����������ж�ÿ�������������ڶ���ʽ�ֲ������������ֲ�
				for (int wIndex = 0; wIndex < docSize; wIndex++) {
					int word = document.get(wIndex);
					//�����Ƕ�s_{di}�ĳ��������õ���ֱ�Ӽ��㣬��û��ʹ�����̶�
					if (lambda * 1/vocabularySize > (1 - lambda)
							* 1/vocabularySize ) {
						//���������������ͳ��
						topicWordCountLF[topic][word] += 1;
						sumTopicWordCountLF[topic] += 1;
						z_words[dIndex][wIndex]=1;
					}else {
						//���Զ���ʽ�ֲ������ͳ��
						topicWordCountDMM[topic][word] += 1;
						sumTopicWordCountDMM[topic] += 1;
						z_words[dIndex][wIndex]=0;
					}
					optimizeTopicVectorsKK(topic);  //�����������������Ҫ�Ż�һ��
				}
				
			}
		}
	}
	//��ʼ������
	public void sampleSingleInitialIteration()
	{
		//��ÿƪ�ĵ�ѭ��
		for (int dIndex = 0; dIndex < numDocuments; dIndex++) {
			//��ȡ�ĵ�
			List<Integer> document = corpus.get(dIndex);
			//�ĵ��ĳ��ȣ����ĵ����������е�����
			int docSize = document.size();
			//�ĵ�������䣬������Ҫ���һ��
			int topic = z[dIndex];
			//�������ɵ��ĵ�ͳ�ƣ��Ƴ����ĵ�
			docTopicCount[topic] = docTopicCount[topic] - 1;
			//ѭ���ĵ���ÿһ������
			for (int wIndex = 0; wIndex < docSize; wIndex++) {
				//��ȡ���ʵı��
				int word = document.get(wIndex);
				//��ȡsubtopic
				int subtopic = z_words[dIndex][wIndex];
				//���subtopic��topic��ͬ�������������������������ڶ���ʽ�ֲ�
				if (subtopic == 1) {
					//����-���� ��Ŀ��1
					topicWordCountLF[topic][word] -= 1;
					//�����Ӧ���ܵĵ�����-1
					sumTopicWordCountLF[topic] -= 1;
				}
				else {
					//����-���� ��Ŀ��1
					topicWordCountDMM[topic][word] -= 1;
					//�����Ӧ���ܵĵ�����-1
					sumTopicWordCountDMM[topic] -= 1;
				}
			}
			// ��ȡ�ĵ����������⣬�����ƪ�ĵ�����ÿ������ĸ��ʣ�Ȼ��������̶Ľ���ѡ��
			for (int tIndex = 0; tIndex < numTopics; tIndex++) {
				//���������ʽ������������,��������Ū�ôʶ��������ڶ���ʽ�ֲ�
				multiPros[tIndex] = (docTopicCount[tIndex] + alpha);
				for (int wIndex = 0; wIndex < docSize; wIndex++) {
					int word = document.get(wIndex);
					multiPros[tIndex] *= (lambda * (topicWordCountLF[tIndex][word] + beta)
							/ (sumTopicWordCountLF[tIndex] + betaSum) + (1 - lambda)
							* (topicWordCountDMM[tIndex][word] + beta)
							/ (sumTopicWordCountDMM[tIndex] + betaSum));
				}
			}
			//�������̶Ľ���ѡ��
			topic = FuncUtils.nextDiscrete(multiPros);
			//�������Ӧ���ĵ�������1
			docTopicCount[topic] ++;;
			//�жϸ����������������������Ƕ���ʽ�ֲ�
			for (int wIndex = 0; wIndex < docSize; wIndex++) {
				int word = document.get(wIndex);// wordID
				//�����Ƕ�s_{di}�ĳ��������õ���ֱ�Ӽ��㣬��û��ʹ�����̶�
				if (lambda * (topicWordCountLF[topic][word] + beta)
						/ (sumTopicWordCountLF[topic] + betaSum) > (1 - lambda)
						* (topicWordCountDMM[topic][word] + beta)
						/ (sumTopicWordCountDMM[topic] + betaSum)) {
					topicWordCountLF[topic][word] += 1;
					sumTopicWordCountLF[topic] += 1;
					z_words[dIndex][wIndex] = 1;
				}
				else {
					topicWordCountDMM[topic][word] += 1;
					sumTopicWordCountDMM[topic] += 1;
					z_words[dIndex][wIndex] = 0;
				}
				z[dIndex] = topic; //��������
			}
		}
	}

	public void writeParameters()
			throws IOException
	{
		BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath + expName + ".paras"));
		writer.write("-model" + "\t" + "LFDMM");
		writer.write("\n-corpus" + "\t" + corpusPath);
		writer.write("\n-vectors" + "\t" + vectorFilePath);
		writer.write("\n-ntopics" + "\t" + numTopics);
		writer.write("\n-alpha" + "\t" + alpha);
		writer.write("\n-beta" + "\t" + beta);
		writer.write("\n-lambda" + "\t" + lambda);
		writer.write("\n-initers" + "\t" + numInitIterations);
		writer.write("\n-niters" + "\t" + numIterations);
		writer.write("\n-twords" + "\t" + topWords);
		writer.write("\n-name" + "\t" + expName);
		if (tAssignsFilePath.length() > 0)
			writer.write("\n-initFile" + "\t" + tAssignsFilePath);
		if (savestep > 0)
			writer.write("\n-sstep" + "\t" + savestep);

		writer.close();
	}

	public void writeDictionary()
			throws IOException
	{
		BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath + expName
				+ ".vocabulary"));
		for (String word : word2IdVocabulary.keySet()) {
			writer.write(word + " " + word2IdVocabulary.get(word) + "\n");
		}
		writer.close();
	}

	public void writeIDbasedCorpus()
			throws IOException
	{
		BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath + expName
				+ ".IDcorpus"));
		for (int dIndex = 0; dIndex < numDocuments; dIndex++) {
			int docSize = corpus.get(dIndex).size();
			for (int wIndex = 0; wIndex < docSize; wIndex++) {
				writer.write(corpus.get(dIndex).get(wIndex) + " ");
			}
			writer.write("\n");
		}
		writer.close();
	}

	public void writeTopicAssignments()
			throws IOException
	{
		BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath + expName
				+ ".topicAssignments"));
		for (int dIndex = 0; dIndex < numDocuments; dIndex++) {
			int docSize = corpus.get(dIndex).size();
			for (int wIndex = 0; wIndex < docSize; wIndex++) {
				writer.write(z_words[dIndex][wIndex]+ " ");
			}
			writer.write("\n");
		}
		writer.close();
	}

	public void writeTopicVectors()
			throws IOException
	{
		BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath + expName
				+ ".topicVectors"));
		for (int i = 0; i < numTopics; i++) {
			for (int j = 0; j < vectorSize; j++)
				writer.write(topicVectors[i][j] + " ");
			writer.write("\n");
		}
		writer.close();
	}

	public void writeTopTopicalWords()
			throws IOException
	{
		BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath + expName
				+ ".topWords"));

		for (int tIndex = 0; tIndex < numTopics; tIndex++) {
			writer.write("Topic" + new Integer(tIndex) + ":");

			Map<Integer, Double> topicWordProbs = new TreeMap<Integer, Double>();
			for (int wIndex = 0; wIndex < vocabularySize; wIndex++) {
				//��ȡ����ֵ��������Կ������������ֵ����ݣ�����������Ϣ�����ں���
				double pro = lambda * expDotProductValues[tIndex][wIndex] / sumExpValues[tIndex]
						+ (1 - lambda) * (topicWordCountDMM[tIndex][wIndex] + beta)
						/ (sumTopicWordCountDMM[tIndex] + betaSum);

				topicWordProbs.put(wIndex, pro);
			}
			//����ʷֲ���������
			topicWordProbs = FuncUtils.sortByValueDescending(topicWordProbs);

			Set<Integer> mostLikelyWords = topicWordProbs.keySet();
			int count = 0;
			for (Integer index : mostLikelyWords) {
				if (count < topWords) {
					writer.write(" " + id2WordVocabulary.get(index));
					count += 1;
				}
				else {
					writer.write("\n\n");
					break;
				}
			}
		}
		writer.close();
	}

	public void writeTopicWordPros()
			throws IOException
	{
		BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath + expName + ".phi"));
		for (int t = 0; t < numTopics; t++) {
			for (int w = 0; w < vocabularySize; w++) {
				double pro = lambda * expDotProductValues[t][w] / sumExpValues[t] + (1 - lambda)
						* (topicWordCountDMM[t][w] + beta) / (sumTopicWordCountDMM[t] + betaSum);
				writer.write(pro + " ");
			}
			writer.write("\n");
		}
		writer.close();
	}

	public void writeDocTopicPros()
			throws IOException
	{
		multiPros = new double[numTopics];
		BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath + expName + ".theta"));
		for (int i = 0; i < numDocuments; i++) {
			int docSize = corpus.get(i).size();
			double sum = 0.0;
			for (int tIndex = 0; tIndex < numTopics; tIndex++) {
				multiPros[tIndex] = (docTopicCount[tIndex] + alpha);
				for (int wIndex = 0; wIndex < docSize; wIndex++) {
					int word = corpus.get(i).get(wIndex);
					multiPros[tIndex] *= (lambda * expDotProductValues[tIndex][word]
							/ sumExpValues[tIndex] + (1 - lambda)
							* (topicWordCountDMM[tIndex][word] + beta)
							/ (sumTopicWordCountDMM[tIndex] + betaSum));
				}
				sum += multiPros[tIndex];
			}
			for (int tIndex = 0; tIndex < numTopics; tIndex++) {
				writer.write((multiPros[tIndex] / sum) + " ");
			}
			writer.write("\n");

		}
		writer.close();
	}
	/**
	 * compute theta
	 * �ϸ������Ĺ�ʽ����
	 */
	public void writerTheta()
			throws IOException
	{
		BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath + expName
				+ ".topictheta"));
		multiPros = new double[numTopics];
		for (int tIndex = 0; tIndex < numTopics; tIndex++) {
			multiPros[tIndex] =  (docTopicCount[tIndex] + alpha)/(numDocuments + numTopics*alpha);
			writer.write(tIndex+"\t"+multiPros[tIndex]+"\n");
		}
		writer.close();
	}
	public void writerIterTopic()
			throws IOException
	{
		BufferedWriter writer = new BufferedWriter(new FileWriter(folderPath + expName
				+ ".itertopic"));
		for (int iter = 0; iter < iter_topic.size(); iter++) {
			writer.write(iter_topic.get(iter) + "\r\n");
		}
		writer.close();
	}
	public void write()
			throws IOException
	{
		//����ʷֲ�
		writeTopTopicalWords();
		/*//�ĵ�����ֲ�,����Ǹ����ļ�����Ҫ��
		writeDocTopicPros();*/
		//����������
		writeTopicAssignments();
		//����ʸ���
		writeTopicWordPros();
		//����ֲ�
		writerTheta();
		//ÿһ��������Ŀ
		writerIterTopic();
	}

	public static void main(String args[])
			throws Exception
	{
		//��ʼ����������----ģ�͵�������
		LFEPDP_CA lfdpmm = new LFEPDP_CA("test/cartest", "test/wordVectors2.txt", 1, 0.1, 0.01, 0.3, 20,
				200, 40, "LFDPMM_Competitor");
		lfdpmm.writeParameters();
		lfdpmm.inference();
	}
}
