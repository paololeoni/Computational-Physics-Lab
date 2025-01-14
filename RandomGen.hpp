#ifndef __RANDOMGEN_H__
#define __RANDOMGEN_H__

#include <cmath>

class RandomGen{

  public:

     RandomGen(unsigned int a, unsigned int c, unsigned int m){
			 m_a = a;
			 m_c = c;
			 m_m = m;
		 };

     RandomGen(unsigned int s){
			 m_a = 1664525;
			 m_c = 1013904223;
			 m_m = pow(2., 31);
			 m_seed = s;
		 };

     void SetA(unsigned int a) {m_a = a;}
     void SetC(unsigned int c) {m_c = c;}
     void SetM(unsigned int m) {m_m = m;}

     double Rand(){

			 unsigned int x = (m_a*m_seed + m_c)%(m_m);
			 m_seed = x;
			 return (x/static_cast<double>(m_m));
			 
		 };

     double Unif(double xmin, double xmax){

			 double r = Rand();
			 double y = xmin + (xmax-xmin)*r;
			 
			 return y;
			 
		 };

     double Exp(double mean){
			 return -log(1-Rand())/mean;
		 };

     double Gauss(double mean, double sigma){
			 
			 double s = Rand();
			 double t = Rand();
			 double x = sqrt(-2.*log(1.-s))*cos(2.*M_PI*t);

			 return mean + x*sigma;
			 
		 };
     double ConstrainedGauss(double mean, double sigma, double sum, int step) {
        double generatedValue = 0;
        do {
            generatedValue = Gauss(mean, sigma);
        } while (std::abs(sum + generatedValue) > (mean + step * sigma) || std::abs(sum + generatedValue) < (mean - step * sigma));
        return generatedValue;
    }
	
     double GaussAR(double mean, double sigma){
			 
			 double xmin = mean -5.*sigma;
			 double xmax = mean + 5.*mean;
			 double fmax = 1./(sqrt(2.*M_PI)*sigma);
			 double x = 0;
			 double y = 0;

			 do{
				 x = Unif(xmin, xmax);
				 y = Unif(0, fmax);
			 } while(y> fmax*exp((-pow(x - mean, 2))/(2.*sigma*sigma)));

			 return x;
			 
		 };

  private:
     unsigned int m_a, m_c, m_m;
     unsigned int m_seed;
};

#endif