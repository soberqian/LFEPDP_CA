package models;

import cc.mallet.optimize.Optimizable;
import cc.mallet.types.MatrixOps;

/**
 * Implementation of the MAP estimation for learning topic vectors, as described
 * in section 3.5 in:
 * 
 * Dat Quoc Nguyen, Richard Billingsley, Lan Du and Mark Johnson. 2015.
 * Improving Topic Models with Latent Feature Word Representations. Transactions
 * of the Association for Computational Linguistics, vol. 3, pp. 299-313.
 * 
 * @author Dat Quoc Nguyen
 */

public class TopicVectorOptimizer
	implements Optimizable.ByGradientValue
{
	// Number of times a word type assigned to the topic
	int[] wordCount;
	int totalCount; // Total number of words assigned to the topic
	int vocaSize; // Size of the vocabulary
	// wordCount.length = wordVectors.length = vocaSize
	double[][] wordVectors;// Vector representations for words
	double[] topicVector;// Vector representation for a topic
	int vectorSize; // vectorSize = topicVector.length

	// For each i_{th} element of topic vector, compute:
	// sum_w wordCount[w] * wordVectors[w][i]
	double[] expectedCountValues;

	double l2Constant; // L2 regularizer for learning topic vectors
	double[] dotProductValues;
	double[] expDotProductValues;
	//主题向量优化
	public TopicVectorOptimizer(double[] inTopicVector, int[] inWordCount,
		double[][] inWordVectors, double inL2Constant)
	{
		//获取所有单词的数
		vocaSize = inWordCount.length;
		//获取词向量的维度
		vectorSize = inWordVectors[0].length;
		//获取L2正则化的值
		l2Constant = inL2Constant;
		//单个主题的向量表示
		topicVector = new double[vectorSize];
		//实现数组的复制（实现上面传入的数组完全复制给topicVector） --参数对应的为：源数组，源数组要复制的起始位置，目标数组，目标数组放置的起始位置，复制的长度
		System.arraycopy(inTopicVector, 0, topicVector, 0, inTopicVector.length);
		//每个词分配到该主题的频数
		wordCount = new int[vocaSize];
		//实现数组的完全复制
		System.arraycopy(inWordCount, 0, wordCount, 0, vocaSize);
		//实现词向量的复制
		wordVectors = new double[vocaSize][vectorSize];
		for (int w = 0; w < vocaSize; w++)
			System
				.arraycopy(inWordVectors[w], 0, wordVectors[w], 0, vectorSize);
		//总单词的统计(包含在一个主题中重复出现的词)
		totalCount = 0;
		for (int w = 0; w < vocaSize; w++) {
			totalCount += wordCount[w];
		}
		//求导后的第一项
		expectedCountValues = new double[vectorSize];
		for (int i = 0; i < vectorSize; i++) {
			for (int w = 0; w < vocaSize; w++) {
				expectedCountValues[i] += wordCount[w] * wordVectors[w][i];
			}
		}

		dotProductValues = new double[vocaSize];
		expDotProductValues = new double[vocaSize];
	}

	@Override
	public int getNumParameters()
	{
		return vectorSize;
	}

	@Override
	public void getParameters(double[] buffer)
	{
		for (int i = 0; i < vectorSize; i++)
			buffer[i] = topicVector[i];
	}

	@Override
	public double getParameter(int index)
	{
		return topicVector[index];
	}

	@Override
	public void setParameters(double[] params)
	{
		for (int i = 0; i < params.length; i++)
			topicVector[i] = params[i];
	}

	@Override
	public void setParameter(int index, double value)
	{
		topicVector[index] = value;
	}
	//计算梯度
	@Override
	public void getValueGradient(double[] buffer)
	{
		double partitionFuncValue = computePartitionFunction(dotProductValues,
			expDotProductValues);

		for (int i = 0; i < vectorSize; i++) {
			buffer[i] = 0.0;

			double expectationValue = 0.0;
			for (int w = 0; w < vocaSize; w++) {
				expectationValue += wordVectors[w][i] * expDotProductValues[w];
			}
			expectationValue = expectationValue / partitionFuncValue;

			buffer[i] = expectedCountValues[i] - totalCount * expectationValue
				- 2 * l2Constant * topicVector[i];
		}
	}
	//计算L_t
	@Override
	public double getValue()
	{
		//计算l_t的对数值
		double logPartitionFuncValue = Math.log(computePartitionFunction(
			dotProductValues, expDotProductValues));

		double value = 0.0;
		//对每个词进行循环
		for (int w = 0; w < vocaSize; w++) {
			if (wordCount[w] == 0)
				continue;
			//分配到该主题的词统计K_{t}{w}*(tau_t*w_w)
			value += wordCount[w] * dotProductValues[w];
		}
		value = value - totalCount * logPartitionFuncValue - l2Constant
			* MatrixOps.twoNormSquared(topicVector);

		return value;
	}

	// Compute the partition function  计算求和函数exp的
	public double computePartitionFunction(double[] elements1,
		double[] elements2)
	{
		//初始值为0
		double value = 0.0;
		//对每个词进行循环
		for (int w = 0; w < vocaSize; w++) {
			elements1[w] = MatrixOps.dotProduct(wordVectors[w], topicVector);
			elements2[w] = Math.exp(elements1[w]);
			value += elements2[w];
		}
		return value;
	}
}
