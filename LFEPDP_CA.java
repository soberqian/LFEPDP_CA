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
 * When coding algorithm algorithm, we also reference the code of the LFDMM that was written by Dat Quoc Nguyen.
 * 
 * @author Yang Qian
 */

public class LFEPDP_CA
{
	public List<String> iter_topic;
	public double alpha; // Hyper-parameter alpha  超参数
	public double beta; // Hyper-parameter beta   超参数
	// public double alphaSum; // alpha * numTopics
	public double betaSum; // beta * vocabularySize  V*beta

	public int numTopics; // Number of topics  主题数目
	public int topWords; // Number of most probable words for each topic  每个主题取多少个靠前的单词

	public double lambda; // Mixture weight value  混合权重值
	public int numInitIterations;  //
	public int numIterations; // Number of EM-style sampling iterations  迭代次数

	public List<List<Integer>> corpus; // Word ID-based corpus  语料单词的id
	public int[] z; //文档对应的主题
	public int[][] z_words; //文档中每个单词对应的子主题，第二维采用0-1表示，0表示来自于多项式分布，1表示来自于隐特征
	// in the corpus
	public int numDocuments; // Number of documents in the corpus  文档的数量
	public int numWordsInCorpus; // Number of words in the corpus  整个语料单词的数量

	public HashMap<String, Integer> word2IdVocabulary; // Vocabulary to get ID  单词的编号
	// given a word
	public HashMap<Integer, String> id2WordVocabulary; // Vocabulary to get word  将编号转化为单词  用于输出
	// given an ID
	public int vocabularySize; // The number of word types in the corpus  整个语料中单词的总数

	// Number of documents assigned to a topic  分配到一个主题文档的数量
	public int[] docTopicCount;
	// numTopics * vocabularySize matrix
	// Given a topic: number of times a word type generated from the topic by
	// the Dirichlet multinomial component  主题对应的单词数量  该单词是由多项式分布产生
	public int[][] topicWordCountDMM;
	// Total number of words generated from each topic by the Dirichlet
	// multinomial component 主题对应的总的单词数量  这些单词是由多项式分布产生
	public int[] sumTopicWordCountDMM;
	// numTopics * vocabularySize matrix
	// Given a topic: number of times a word type generated from the topic by
	// the latent feature component  单词是由隐特征部分产生  统计一个主题对应的单词数量
	public int[][] topicWordCountLF;
	// Total number of words generated from each topic by the latent feature
	// component  单词由隐特征产生  统计一个主题对应的总单词数量
	public int[] sumTopicWordCountLF;
	// Double array used to sample a topic   概率 用于抽样
	public double[] multiPros;
	// Path to the directory containing the corpus  
	public String folderPath;
	// Path to the topic modeling corpus
	public String corpusPath;
	public String vectorFilePath;
	public double[][] wordVectors; // Vector representations for words  词向量表示
	public double[][] topicVectors;// Vector representations for topics 主题向量表示
	public int vectorSize; // Number of vector dimensions  向量的维度
	public double[][] dotProductValues;   //点乘法的值
	public double[][] expDotProductValues;  //指数变化后的值
	public double[] sumExpValues; // Partition function values 求和的值

	public final double l2Regularizer = 0.01; // L2 regularizer value for learning topic vectors L2正则化
	public final double tolerance = 0.05; // Tolerance value for LBFGS convergence  LBFGS收敛
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
	//这里初始化
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
		//word2vec语料
		vectorFilePath = pathToWordVectorsFile;
		//语料的路径
		corpusPath = pathToCorpus;
		folderPath = pathToCorpus.substring(0,
				Math.max(pathToCorpus.lastIndexOf("/"), pathToCorpus.lastIndexOf("\\")) + 1);
		//输入语料的路径
		System.out.println("Reading topic modeling corpus: " + pathToCorpus);
		//词转化为编号
		word2IdVocabulary = new HashMap<String, Integer>();
		//编号转化为词
		id2WordVocabulary = new HashMap<Integer, String>();
		//语料
		corpus = new ArrayList<List<Integer>>();
		//文档数目
		numDocuments = 0;
		//语料中单词的数目
		numWordsInCorpus = 0;
		//读取语料
		BufferedReader br = null;
		try {
			int indexWord = -1;
			br = new BufferedReader(new FileReader(pathToCorpus));
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
				//文档数目++
				numDocuments++;
				//语料中所有单词的数量
				numWordsInCorpus += document.size();
				//将所有文档添加到集合中
				corpus.add(document);
			}
		}
		catch (Exception e) {
			e.printStackTrace();
		}
		//语料不重复单词的总量
		vocabularySize = word2IdVocabulary.size();
		//主题对应的文档统计
		docTopicCount = new int[numTopics];
		//主题-单词统计  来自多项式分布
		topicWordCountDMM = new int[numTopics][vocabularySize];
		//主题对应的单词总数目统计 来自多项式分布
		sumTopicWordCountDMM = new int[numTopics];
		//主题-单词统计 来自于隐特征分布
		topicWordCountLF = new int[numTopics][vocabularySize];
		//主题对应的单词数目总计 来自于隐特征
		sumTopicWordCountLF = new int[numTopics];
		//多项式分布的先验
		multiPros = new double[numTopics];
		//先验为1/K,后面要轮盘赌的，在初始化的时候
		for (int i = 0; i < numTopics; i++) {
			multiPros[i] = 1.0 / numTopics;
		}

		// alphaSum = numTopics * alpha;  
		betaSum = vocabularySize * beta;  
		//读取词向量 word2vec文件
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
	//读取词向量文件
	public void readWordVectorsFile(String pathToWordVectorsFile)
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
	//初始化方法
	public void initialize()
			throws IOException
	{
		//随机对文档进行主题分配
		System.out.println("Randomly initialzing topic assignments ...");
		z = new  int[numDocuments];
		z_words = new  int[numDocuments][];  //初始化
		//循环每篇文档
		for (int docId = 0; docId < numDocuments; docId++) {
			//基于轮盘赌获取主题编号（前面已经初始化了），这里multiPros必须有值，否则则会报错
			int topic= FuncUtils.nextDiscrete(multiPros);
			z[docId] = topic; //赋值主题
			//分配到该主题的文档数量+1
			docTopicCount[topic] += 1;
			//文档的单词个数
			int docSize = corpus.get(docId).size();
			z_words[docId] = new int[docSize];  //对每篇文档又来了一次初始化，这个要习惯使用它
			//循环每个单词
			for (int j = 0; j < docSize; j++) {
				//获取单词编号
				int wordId = corpus.get(docId).get(j);
				//随机产生false or true,用来初始化该文档是来自于隐特征还是多项式分布
				boolean component = new Randoms().nextBoolean();
				if (!component) { // Generated from the latent feature component
					//主题-单词数量增加1  由隐特征主题生成
					topicWordCountLF[topic][wordId] += 1;
					// 该主题生成的单词总数增加1 由隐特征主题生成
					sumTopicWordCountLF[topic] += 1;
					z_words[docId][j] = 1;//1表示来自于隐特征分布
				}
				else {// Generated from the Dirichlet multinomial component
					//主题-单词数量增加1 由多项式分布生成
					topicWordCountDMM[topic][wordId] += 1;
					//主题生成的单词总数增加1
					sumTopicWordCountDMM[topic] += 1;
					z_words[docId][j] = 0; //表示来自于多项式分布
				}
			}
		}
	}
	//模型推断
	public void inference()
			throws IOException
	{
		System.out.println("Running Gibbs sampling inference: ");
		//初始化迭代
		for (int iter = 1; iter <= numInitIterations; iter++) {

			System.out.println("\tInitial sampling iteration: " + (iter));
			//单词初始化迭代
			sampleSingleInitialIteration();
		}
		optimizeTopicVectors();
		//正式迭代
		for (int iter = 1; iter <= numIterations; iter++) {
			//			intialize_cluster(docTopicCount,topicWordCountDMM,topicWordCountLF,sumTopicWordCountDMM,sumTopicWordCountLF);
			iter_topic.add(iter + "\t" + numTopics);
			System.out.println("\tLFDMM sampling iteration: " + (iter)+"\tnumber of topic:"+numTopics);
			optimizeTopicVectors();
			//抽取主题
			sampleSingleIteration();
			defragment();
			if ((savestep > 0) && (iter % savestep == 0) && (iter < numIterations)) {
				System.out.println("\t\tSaving the output from the " + iter + "^{th} sample");
				expName = orgExpName + "-" + iter;
				write();
			}
		}
		expName = orgExpName;
		//保存模型相关参数
		writeParameters();
		System.out.println("Writing output from the last sample ...");
		//保存信息
		write();

		System.out.println("Sampling completed!");
	}
	//优化主题向量
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
						//主题向量表示  主题包含的单词个数  词向量 正则化值(这里是传入参数-----以便执行TopicVectorOptimizer)
						TopicVectorOptimizer optimizer = new TopicVectorOptimizer(
								topicVectors[topic], topicWordCountLF[topic], wordVectors, l2Value);
						//通过LBFGS优化
						Optimizer gd = new LBFGS(optimizer, tolerance);
						gd.optimize(600);
						//需要优化的参数
						optimizer.getParameters(topicVectors[topic]);
						//输入的是两个特征的乘积以及其加和-----针对每个主题计算一个向量值（为了更新主题使用）
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
	//优化某一个主题的向量
	public  void optimizeTopicVectorsKK(int k)
	{
		System.out.println("\t\tEstimating new topic vectors ...");
		int rate = 1;
		boolean check = true;
		while (check) {
			double l2Value = l2Regularizer * rate;
			try {
				//主题向量表示  主题包含的单词个数  词向量 正则化值(这里是传入参数-----以便执行TopicVectorOptimizer)
				TopicVectorOptimizer optimizer = new TopicVectorOptimizer(
						topicVectors[k], topicWordCountLF[k], wordVectors, l2Value);
				//通过LBFGS优化
				Optimizer gd = new LBFGS(optimizer, tolerance);
				gd.optimize(600);
				//需要优化的参数
				optimizer.getParameters(topicVectors[k]);
				//输入的是两个特征的乘积以及其加和-----针对每个主题计算一个向量值（为了更新主题使用）
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
	 * @param 输入参数为(1)主题对应的文档数量统计;(2)主题包含单词的统计，这里包括主题包含的隐特征和多项式分布生成（3）主题包含的总单词统计，包括多项式
	 * 分布生成和隐特征分布生成
	 * @return
	 * @Date: 2018-3-29
	 * @Author: yang qian 
	 * @Description: remove a topic
	 */
	//移除主题,主题重新编号，进行重新分配------这里主题向量的优化必须也在这里更新
	public void intialize_cluster(int[] docTopicCount,int[][] topicWordCountDMM,int[][] topicWordCountLF, 
			int[] sumTopicWordCountDMM, int[] sumTopicWordCountLF){
		//主题重新分配
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
		//主题数目会发生变化
		numTopics=countk.keySet().size();
		//		System.out.println(numTopics);
		//主题对应的文档数量
		for(int i=0;i<numDocuments;i++){
			z[i]=countk.get(z[i]); //Map的value值
		}
		this.docTopicCount = new int [numTopics]; //主题对应的文档数量
		this.topicWordCountDMM = new int [numTopics][vocabularySize]; //主题z对应的单词v的数量，来自于多项式分布
		this.topicWordCountLF = new int [numTopics][vocabularySize];; //主题z对应的单词v的数量，来自于多项式分布
		this.sumTopicWordCountDMM = new int[numTopics];
		this.sumTopicWordCountLF = new int[numTopics];
		//循环每个主题，做相关初始化赋值
		for(int k = 0; k < numTopics; k++){
			this.docTopicCount[k] = 0;
			this.sumTopicWordCountDMM[k] = 0;
			this.sumTopicWordCountLF[k] = 0;
			for(int t = 0; t < vocabularySize; t++){
				this.topicWordCountDMM[k][t] = 0;
				this.topicWordCountLF[k][t] = 0;
			}
		}
		//重新统计值
		for(int dIndex = 0; dIndex < numDocuments; dIndex++){
			List<Integer> document = corpus.get(dIndex); 
			int docSize = document.size();  //获取文档的单词数量
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
		//重新初始化,主题向量
		this.topicVectors = new double[numTopics][vectorSize];
	}
	/**
	 * @param 将主题中不包含文档的主题移除
	 * @return
	 * @Date: 2018-3-29
	 * @Author: yang qian 
	 * @Description: remove a topic
	 */
	public void removeDocument(int docID){
		//获取文档的所有单词
		List<Integer> document = corpus.get(docID);
		//文档的长度
		int docSize = document.size();
		//文档的初始主题分布，接下来是移除该单词
		int topic = z[docID];
		//主题对应的文档数目减1
		docTopicCount[topic] = docTopicCount[topic] - 1;
		for (int wIndex = 0; wIndex < docSize; wIndex++) {
			//获取单词的id
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
		//这里还要重新处理文档对应的主题
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
	//将数组放大，确保不越界
	public  int[] ensureCapacity(int[] arr,int i) {
		int length = arr.length;
		int[] arr2 = new int[length+i];
		System.arraycopy(arr, 0, arr2, 0, length);
		return arr2;
	}
	public  int[][] ensureCapacity(int[][] array,int i,int j) {  
		int[][] arr = new int[array.length +i][array[0].length +j];       //扩展  
		for(int c = 0; c< array.length; c++) {  
			System.arraycopy(array[c], 0, arr[c], 0, array[c].length);  //数组拷贝  
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
		double[][] arr = new double[array.length +i][array[0].length +j];       //扩展  
		for(int c = 0; c< array.length; c++) {  
			System.arraycopy(array[c], 0, arr[c], 0, array[c].length);  //数组拷贝  
		}  
		return arr;  
	} 
	//每一代分配主题
	public void sampleSingleIteration()
	{
		//对每一篇文档进行循环
		for (int dIndex = 0; dIndex < numDocuments; dIndex++) {
			//			System.out.println("当前执行的文档是："+dIndex);
			double[] prob = new double[numTopics+1];
			//获取文档的所有单词
			List<Integer> document = corpus.get(dIndex);
			//文档的长度
			int docSize = document.size();
			//文档的初始主题分布，接下来是移除该单词
			int topic = z[dIndex];
			//主题对应的文档数目减1
			docTopicCount[topic] = docTopicCount[topic] - 1;
			if (docTopicCount[topic] < 0) {
				System.out.println("docTopicCount < 0 "+topic +" " +docTopicCount[topic]);
				defragment();
			}
			//接下来对对个单词进行循环，做相关单词的统计工作
			for (int wIndex = 0; wIndex < docSize; wIndex++) {
				//获取单词的id
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
			// 对文档单词的主题进行抽样,计算概率，这里需要主题新主题产生的概率怎么计算
			for (int tIndex = 0; tIndex < numTopics; tIndex++) {
				prob[tIndex] = docTopicCount[tIndex]/(numDocuments - 1 + alpha);
				for (int wIndex = 0; wIndex < docSize; wIndex++) {
					int word = document.get(wIndex);
					//依据公式进行计算，不过论文公式有问题  N_{d,w}+K_{d,w}的次方有问题，推理的公式应该是这样的
					prob[tIndex] *= (lambda * expDotProductValues[tIndex][word]
							/ sumExpValues[tIndex] + (1 - lambda)
							* (topicWordCountDMM[tIndex][word] + beta)
							/ (sumTopicWordCountDMM[tIndex] + betaSum));
				}
			}

			prob[numTopics]= (alpha) / (numDocuments - 1 + alpha);
			double valueOfRule = 1.0;
			//对每个单词进行循环
			for(int wIndex = 0; wIndex < docSize; wIndex++){
				valueOfRule *= 1.0/vocabularySize;  //如果是新主题会发现隐特征和多项式分布生成的概率是一样的
			}
			prob[numTopics] = prob[numTopics] * valueOfRule ;
			//基于轮盘赌选择是已有的簇还是旧的簇
			topic = FuncUtils.nextDiscrete(prob);  //轮盘赌没有问题为什么会出现越界的情况呢
			z[dIndex] = topic;  //文档对应的新主题
			//判断该主题是否为新主题
			if(topic < numTopics){
				docTopicCount[topic] += 1;
				//开始做相关统计
				for (int wIndex = 0; wIndex < docSize; wIndex++) {
					int word = document.get(wIndex);
					//这里是对s_{di}的抽样，采用的是直接计算，并没有使用轮盘赌
					if (lambda * expDotProductValues[topic][word] / sumExpValues[topic] > (1 - lambda)
							* (topicWordCountDMM[topic][word] + beta)
							/ (sumTopicWordCountDMM[topic] + betaSum)) {
						//来自隐特征的相关统计
						topicWordCountLF[topic][word] += 1;
						sumTopicWordCountLF[topic] += 1;
						z_words[dIndex][wIndex]=1;
					}
					else {
						//来自多项式分布的相关统计
						topicWordCountDMM[topic][word] += 1;
						sumTopicWordCountDMM[topic] += 1;
						z_words[dIndex][wIndex]=0;
					}
				}
			}else if (topic == numTopics) {   // a new topic is created
				numTopics++;  //主题数量增多
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
				//这里新主题判断每个单词是来自于多项式分布还是隐特征分布
				for (int wIndex = 0; wIndex < docSize; wIndex++) {
					int word = document.get(wIndex);
					//这里是对s_{di}的抽样，采用的是直接计算，并没有使用轮盘赌
					if (lambda * 1/vocabularySize > (1 - lambda)
							* 1/vocabularySize ) {
						//来自隐特征的相关统计
						topicWordCountLF[topic][word] += 1;
						sumTopicWordCountLF[topic] += 1;
						z_words[dIndex][wIndex]=1;
					}else {
						//来自多项式分布的相关统计
						topicWordCountDMM[topic][word] += 1;
						sumTopicWordCountDMM[topic] += 1;
						z_words[dIndex][wIndex]=0;
					}
					optimizeTopicVectorsKK(topic);  //如果产生新主题则需要优化一下
				}
				
			}
		}
	}
	//初始化迭代
	public void sampleSingleInitialIteration()
	{
		//对每篇文档循环
		for (int dIndex = 0; dIndex < numDocuments; dIndex++) {
			//获取文档
			List<Integer> document = corpus.get(dIndex);
			//文档的长度，即文档包含的所有单词数
			int docSize = document.size();
			//文档主题分配，这里需要理解一下
			int topic = z[dIndex];
			//主题生成的文档统计，移除该文档
			docTopicCount[topic] = docTopicCount[topic] - 1;
			//循环文档的每一个单词
			for (int wIndex = 0; wIndex < docSize; wIndex++) {
				//获取单词的编号
				int word = document.get(wIndex);
				//获取subtopic
				int subtopic = z_words[dIndex][wIndex];
				//如果subtopic和topic相同，来自于隐变量，否则来自于多项式分布
				if (subtopic == 1) {
					//主题-单词 数目减1
					topicWordCountLF[topic][word] -= 1;
					//主题对应的总的单词数-1
					sumTopicWordCountLF[topic] -= 1;
				}
				else {
					//主题-单词 数目减1
					topicWordCountDMM[topic][word] -= 1;
					//主题对应的总的单词数-1
					sumTopicWordCountDMM[topic] -= 1;
				}
			}
			// 抽取文档所属的主题，计算该篇文档属于每个主题的概率，然后基于轮盘赌进行选择
			for (int tIndex = 0; tIndex < numTopics; tIndex++) {
				//这里这个公式是哪里来的呢,这里作者弄得词都是来自于多项式分布
				multiPros[tIndex] = (docTopicCount[tIndex] + alpha);
				for (int wIndex = 0; wIndex < docSize; wIndex++) {
					int word = document.get(wIndex);
					multiPros[tIndex] *= (lambda * (topicWordCountLF[tIndex][word] + beta)
							/ (sumTopicWordCountLF[tIndex] + betaSum) + (1 - lambda)
							* (topicWordCountDMM[tIndex][word] + beta)
							/ (sumTopicWordCountDMM[tIndex] + betaSum));
				}
			}
			//基于轮盘赌进行选择
			topic = FuncUtils.nextDiscrete(multiPros);
			//新主题对应的文档数量加1
			docTopicCount[topic] ++;;
			//判断该主题是来自于隐特征还是多项式分布
			for (int wIndex = 0; wIndex < docSize; wIndex++) {
				int word = document.get(wIndex);// wordID
				//这里是对s_{di}的抽样，采用的是直接计算，并没有使用轮盘赌
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
				z[dIndex] = topic; //更新主题
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
				//获取概率值，这里可以看出包含两部分的内容，将两部分信息进行融合了
				double pro = lambda * expDotProductValues[tIndex][wIndex] / sumExpValues[tIndex]
						+ (1 - lambda) * (topicWordCountDMM[tIndex][wIndex] + beta)
						/ (sumTopicWordCountDMM[tIndex] + betaSum);

				topicWordProbs.put(wIndex, pro);
			}
			//主题词分布降序排序
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
	 * 严格按照论文公式来的
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
		//主题词分布
		writeTopTopicalWords();
		/*//文档主题分布,这个是个大文件，不要了
		writeDocTopicPros();*/
		//主题分配情况
		writeTopicAssignments();
		//主题词概率
		writeTopicWordPros();
		//主题分布
		writerTheta();
		//每一代主题数目
		writerIterTopic();
	}

	public static void main(String args[])
			throws Exception
	{
		//初始化迭代次数----模型迭代次数
		LFEPDP_CA lfdpmm = new LFEPDP_CA("test/cartest", "test/wordVectors2.txt", 1, 0.1, 0.01, 0.3, 20,
				200, 40, "LFDPMM_Competitor");
		lfdpmm.writeParameters();
		lfdpmm.inference();
	}
}
