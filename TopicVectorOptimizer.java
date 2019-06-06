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
	//���������Ż�
	public TopicVectorOptimizer(double[] inTopicVector, int[] inWordCount,
		double[][] inWordVectors, double inL2Constant)
	{
		//��ȡ���е��ʵ���
		vocaSize = inWordCount.length;
		//��ȡ��������ά��
		vectorSize = inWordVectors[0].length;
		//��ȡL2���򻯵�ֵ
		l2Constant = inL2Constant;
		//���������������ʾ
		topicVector = new double[vectorSize];
		//ʵ������ĸ��ƣ�ʵ�����洫���������ȫ���Ƹ�topicVector�� --������Ӧ��Ϊ��Դ���飬Դ����Ҫ���Ƶ���ʼλ�ã�Ŀ�����飬Ŀ��������õ���ʼλ�ã����Ƶĳ���
		System.arraycopy(inTopicVector, 0, topicVector, 0, inTopicVector.length);
		//ÿ���ʷ��䵽�������Ƶ��
		wordCount = new int[vocaSize];
		//ʵ���������ȫ����
		System.arraycopy(inWordCount, 0, wordCount, 0, vocaSize);
		//ʵ�ִ������ĸ���
		wordVectors = new double[vocaSize][vectorSize];
		for (int w = 0; w < vocaSize; w++)
			System
				.arraycopy(inWordVectors[w], 0, wordVectors[w], 0, vectorSize);
		//�ܵ��ʵ�ͳ��(������һ���������ظ����ֵĴ�)
		totalCount = 0;
		for (int w = 0; w < vocaSize; w++) {
			totalCount += wordCount[w];
		}
		//�󵼺�ĵ�һ��
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
	//�����ݶ�
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
	//����L_t
	@Override
	public double getValue()
	{
		//����l_t�Ķ���ֵ
		double logPartitionFuncValue = Math.log(computePartitionFunction(
			dotProductValues, expDotProductValues));

		double value = 0.0;
		//��ÿ���ʽ���ѭ��
		for (int w = 0; w < vocaSize; w++) {
			if (wordCount[w] == 0)
				continue;
			//���䵽������Ĵ�ͳ��K_{t}{w}*(tau_t*w_w)
			value += wordCount[w] * dotProductValues[w];
		}
		value = value - totalCount * logPartitionFuncValue - l2Constant
			* MatrixOps.twoNormSquared(topicVector);

		return value;
	}

	// Compute the partition function  ������ͺ���exp��
	public double computePartitionFunction(double[] elements1,
		double[] elements2)
	{
		//��ʼֵΪ0
		double value = 0.0;
		//��ÿ���ʽ���ѭ��
		for (int w = 0; w < vocaSize; w++) {
			elements1[w] = MatrixOps.dotProduct(wordVectors[w], topicVector);
			elements2[w] = Math.exp(elements1[w]);
			value += elements2[w];
		}
		return value;
	}
}
