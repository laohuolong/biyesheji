package service;

import java.io.IOException;
import service.*;
/**
 * Created by ecnu105 on 2017/3/13.
 */
public class testSvm {
    public static void main(String[] args) throws IOException{
        //String[] arg ={"-t","0","-w1","2","-w3","2","-w7","2","E:\\data\\pre_train.txt","E:\\result\\model_r_linear_137.txt"};
        String[] arg ={"-t","0","F:\\data\\pre_train_2500.txt","F:\\data\\model_linear2500.txt"};
		//String[] arg ={"-t","0","F:\\data\\pre_train_50.txt","F:\\data\\model_linear50.txt"};
        String[] parg = {"F:\\data\\pre_test_2500.txt","F:\\data\\model_linear2500.txt","F:\\data\\out_linear2500.txt"};
		//String[] parg = {"F:\\data\\pre_test_50.txt","F:\\data\\model_linear50.txt","F:\\data\\out_linear50.txt"};
        System.out.println(".........SVM运行开始.........");
        svm_train t =new svm_train();
        svm_predict p = new svm_predict();
        t.main(arg);
        p.main(parg);
    }
}
