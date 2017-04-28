package libsvm;
public class svm_parameter implements Cloneable,java.io.Serializable
{
	/* svm_type */
	public static final int C_SVC = 0;
	public static final int NU_SVC = 1;
	public static final int ONE_CLASS = 2;
	public static final int EPSILON_SVR = 3;
	public static final int NU_SVR = 4;

	/* kernel_type */
	public static final int LINEAR = 0;//线性核函数
	//linear: K(xi,xj) = xi * xj;
	public static final int POLY = 1;//多项式核函数
	//poly :K(xi,xj) = ( r1 xi * xj + r2 )^d ,r1>0;
	public static final int RBF = 2;//高斯径向基函数
	//rbf: K(xi,xj) = exp( -r1 ||xi -xj||^2) ,r1>0
	public static final int SIGMOID = 3;//sigmod核函数
	//sigmoid:K(xi,xj) = tanh( r1 xi * xj+ r2);
	public static final int PRECOMPUTED = 4;//自定义核函数

	public int svm_type;//向量机优化类型
	public int kernel_type;//核函数的类型
	public int degree;	// for poly 函数次数
	public double gamma;	// for poly/rbf/sigmoid
	public double coef0;	// for poly/sigmoid
	//coef() = r2, 默认为0;
	//degree: 多项式次数,默认为3次;
	//gamma = r1: 默认值为1/k, k为类别数;


	// these are for training only
	public double cache_size; // in MB   缓存大小
	public double eps;	// stopping criteria  停止的条件
	public double C;	// for C_SVC, EPSILON_SVR and NU_SVR  惩罚项系数
	public int nr_weight;		// for C_SVC
	public int[] weight_label;	// for C_SVC
	public double[] weight;		// for C_SVC
	public double nu;	// for NU_SVC, ONE_CLASS, and NU_SVR
	public double p;	// for EPSILON_SVR
	public int shrinking;	// use the shrinking heuristics
	public int probability; // do probability estimates

	public Object clone() 
	{
		try 
		{
			return super.clone();
		} catch (CloneNotSupportedException e) 
		{
			return null;
		}
	}

}
