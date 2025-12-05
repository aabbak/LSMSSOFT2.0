/* The LSMSSOFT2.0 computes a gravimetric geoid by using the KTH method and additive corrections.
 * 
 * This program is distributed under the terms of a GNU License (ver. 1).
 * Hopefully, this program will be useful, but WITHOUT ANY WARRANTY. 
 *
 * Author    : R. Alpay ABBAK
 * Address    : Konya Technical University, Geomatics Engineering Dept, Konya, TURKEY
 * E-Mail     : raabbak@ktun.edu.tr
 *
 * Reference paper for the program:
 * Abbak et al., (2025). A Novel Multithreaded Parallel Computing Approach for Solving 
 * Convolution Integrals in Geoid Modelling, Computer & Geosciences, DOI:  
 *
 * Compilation of the program on Linux: 
 * g++ LSMSSOFT2.0.cpp matris.cpp levelell.cpp -o LSMSSOFT -pthread
 *
 * Execution of the program with our sample data: 
 * ./LSMSSOFT -GITU_GGC16.gfc -Aanomaly.xyz -Eelevation.xyz
 * 
 * Created : 01.01.2014             		// v1.0  
 * Updated : 04.09.2019				// crust density added
 * Updated : 15.05.2021				// simple corrections
 * Updated : 25.11.2025				// parallel computing
 */
#include <time.h>
#include <signal.h>
#include <thread>
#include <iostream>
#include <cstdlib>
#include <pthread.h>
#include <unistd.h>				// Standard option library 
#include<string.h>				// Standard string library
#include"matris.h"				// User-defined matrix library
#include"levelell.h"				// Level ellipsoid computations
#define G 6.672585e-11				// Gravitational constant [m3/kg.s2]
#define	a 6378137.0				// Semimajor axes of GRS80 ellipsoid
#define R 6371000.0				// The Earth's mean radius
#define pi 3.141592653589793238			// Constant pi
#define rho 57.2957795130823209			// 180degree/pi
#define omega 7292115.0e-11			// Angular velocity of the Earth's rotation
#define GM 0.3986005e+15 			// Gravitation constant of level ellipsoid
#define J2 108263.0e-8				// Dynamical form factor of the Earth (GRS80)
#define mx 1200					// Maximum dimension of the data area 
#define Nmax 2000				// Maximum degree of series expansion 
#define OPTIONS "G:A:E:M:D:P:C:V:R:I:SH"	// Options
int nthreads=std::thread::hardware_concurrency()/2;
struct timespec start, finish;
double    elapsed=0.0;
void COEFFICIENT(int M,int v,double C0,matris c,matris dc,matris Q,matris E,matris &Sn,matris &bn);
void LEGENDRE(matris &P,int N,double x);
void HELP();
void gradient_compute(double phistart, double phiend);
void stokes_compute(double phistart, double phiend, double QM0, matris *bnn);
struct thread_data1
{
	int    thread_id;
   	double  phistart;
   	double    phiend;
};
void *thread1(void *threadarg) 
{
   	struct thread_data1 *my_data; 			// verinin tipi struct olmalı
   	my_data=(struct thread_data1 *)threadarg;  	// tip donusturme
   	gradient_compute(my_data->phistart,my_data->phiend);
   	pthread_exit(NULL);
}
struct thread_data2
{
	int    thread_id;
   	double  phistart;
   	double    phiend;
	double       QM0;
	matris 	     *bn;
};
void *thread2(void *threadarg)
{
   	struct thread_data2 *my_data; 			// verinin tipi struct olmalı
   	my_data=(struct thread_data2 *)threadarg;  	// tip donusturme
   	stokes_compute(my_data->phistart,my_data->phiend,my_data->QM0,my_data->bn);
   	pthread_exit(NULL);
}
int           M=115; 			// maximum expansion of GGM used
int  	      v=1;			// type (biased:1 unbiased:2 optimum:3)
int 	      s=0;			// shows if all segments are printed
double 	psi0=1.0/rho;			// spherical cap size in radian
double       C0=16.0;			// variance of terrestrial gravity data
double 	 MinPhi=43.01;			// minimum latitude of target area
double 	 MaxPhi=49.01;			// maximum latitude of target area
double 	 MinLam=0.01;			// minimum longitude of target area
double 	 MaxLam=6.01;			// maximum longitude of target area
double 	 PhiInt=0.02;			// grid size in latitude direction
double 	 LamInt=0.02;			// grid size in longitude direction
double MinDatPhi=0.0;			// minimum latitude of data area
double MinDatLam=0.0;			// minimum longitude of data area
double constant=PhiInt*LamInt/rho/rho;	// grid size of the block 
matris 	C(2000000);			// Cnm coefficients
matris  S(2000000);			// Snm coefficients
matris cs(2000000);			// variance of Cnm
matris ss(2000000);			// variance of Snm
matris  Dgr(mx,mx);			// gravity gradient
matris coslati(mx);			// latitude of running point
matris sinlati(mx);			// latitude of running point
matris    loni(mx);			// longitude of running point
matris   dg(mx,mx);			// terrestrial gravity anomalies
matris    H(mx,mx);			// topographic heights 
matris    D(mx,mx);			// mean crust density 
matris approxN(mx,mx);			// approximate undulation
matris 	 top(mx,mx);			// topographic effect 
matris 	 atm(mx,mx);			// atmospheric effect 
matris 	 dwc(mx,mx);			// downward continuation effect 
matris 	 ell(mx,mx);			// ellipsoidal effect 
matris geoid(mx,mx);			// Final geoid model
FILE   *density=NULL;			// density data file
int main(int argc, char *argv[]) 
{
// DEFINITIONS **************************************************************************
	clock_gettime(CLOCK_MONOTONIC, &start);	
	FILE 	 *model=NULL;			// reference global geopotential model
	FILE   *anomaly=NULL;			// anomaly data file 
	FILE *elevation=NULL;			// elevation data file
	char 	      option;			// temprorary option
    	char       line[256];  			// read a line from model file
    	const char *slash="/";			// discriminant symbol
	int           n=0;			// array element of degree
    	int           m=0;			// array element of order
    	int           i=0;			// array element
    	int           j=0;			// array element
	int           x=0;			// array element
    	int           y=0;			// array element
    	int          in=0;			// array element of computation point
    	int          jn=0;			// array element of computation point
	int        Mmax=0;			// maximum expansion of GGM
	int        mmax=0;			// element of max expansion of GGM
	double     phi=0.0;			// latitude of computational point [deg] 
    	double     lam=0.0;			// longitude of computational point [deg]
	double    phii=0.0;			// latitude of computational point [rad]
    	double    lami=0.0;			// longitude of computational point [rad]
 	double      HC=0.0; 			// Cnm coefficient
    	double      HS=0.0; 			// Snm coefficient
    	double      Hc=0.0; 			// variance of Cnm coefficient
    	double      Hs=0.0; 			// variance of Snm coefficient
	double scale=pow(GM,2)/pow(a,4)*1e+10;  // scale factor for variance in mGal
	double       t=0.0;			// cos(psi)
	double     S_t=0.0;
	double    dS_t=0.0;			// first derivation of Stokes' function
	double      tt=0.0;			// sub-value
	double      K0=0.0;			// "	
	double     QM0=0.0;			// Molodenski coefficient
	double      K1=0.0;			// "
	double      K2=0.0;			// "
	double      I2=0.0;			// "
	double     tot=0.0;			// total value
	double      Q0=0.0;			// Molodenski coefficient
	matris   c(Nmax+2);			// Degree variances
	matris  dc(Nmax+2);			// Error degree variances
	matris  PN(Nmax+9);			// Unnormalized Legendre polynomial
  	matris 	 Q(Nmax+2); 		      	// Molodenski coefficient
	matris   I(Nmax+2);    			// sub-value
	matris   T(Nmax+2);    			// "
	matris   O(Nmax+2);	        	// "
//** OPTION ANALYSIS ***********************************************************************
	if(argc<4) HELP();
	while((option=getopt(argc,argv,OPTIONS))!=-1)
        switch(option)
        {
		case 'G':
			model=fopen(optarg,"r");	
			break;
		case 'A':
			anomaly=fopen(optarg,"r");
			break;
		case 'E':
			elevation=fopen(optarg,"r");	
                	break;
		case 'D':
			density=fopen(optarg,"r");
			if(density==NULL)
    			{
				printf("\nDensity file can not be opened!!!\n\n");
        			exit(EXIT_FAILURE);
			}
                	break;
		case 'M':
			M=atoi(optarg);
			break;
		case 'P':
			psi0=atof(optarg)/rho;
			break;
		case 'C':
			C0=atof(optarg);	
			break;
		case 'V':
			v=atoi(optarg);	
			break;
		case 'R':
			MinPhi=atof(strtok(optarg,slash));
			MaxPhi=atof(strtok(NULL,slash));
			MinLam=atof(strtok(NULL,slash));
			MaxLam=atof(strtok(NULL,slash));
                	break;
		case 'I':
                	PhiInt=atof(strtok(optarg,slash));
                	LamInt=atof(strtok(NULL,slash));
                	break;
		case 'S':
                	s=1;
			break;
		case 'H':
                	HELP();
		default:
                	HELP();
        }
//***** READ FILES and DEFINE LIMITS *******************************************************
	constant=PhiInt*LamInt/rho/rho;	// grid size of the block 
	if(model==NULL)
    	{
		printf("\nGGM file can not be opened!!!\n\n");
        	exit(EXIT_FAILURE);
    	}
	if(anomaly==NULL)
    	{
		printf("\nAnomaly file can not be opened!!!\n\n");
        	exit(EXIT_FAILURE);
    	}
	if(elevation==NULL)
    	{
		printf("\nElevation file can not be opened!!!\n\n");
        	exit(EXIT_FAILURE);
    	}
	while(fgets(line,256,model)!=NULL)
    	{
        	if(sscanf(line,"%i%i%lf%lf%lf%lf",&n,&m,&HC,&HS,&Hc,&Hs)==6)
        	{
       		        i=n*(n+1)/2+m;
                	C(i)=HC; 	S(i)=HS;	cs(i)=Hc;      	ss(i)=Hs;
        	}
    	}
	Mmax=n;
    	fclose(model);
    	if(C(0)==0.0)       C(0)=1.0;
	LevelEllipsoid levell(a,J2,GM,omega);		// Create normal gravity field (GRS80) 
	for(i=0;i<11;i++)		// Compute harmonic coefficients of disturbing potential 
		C(2*i*i+i)-=levell.sphcoefs(i)/sqrt(4*i+1);
	MinDatPhi=MinPhi-4.0;
	MinDatLam=MinLam-4.0;
	while(!feof(anomaly))
	{
		fscanf(anomaly,"%lf%lf",&phi,&lam);
		i=round((phi-MinDatPhi)/PhiInt); 	j=round((lam-MinDatLam)/LamInt);
		coslati(i)=cos(phi/rho); 	
		sinlati(i)=sin(phi/rho); 	
		loni(j)=lam/rho;
		fscanf(anomaly,"%lf\n",&dg(i,j));
    	}
    	fclose(anomaly);
	while(!feof(elevation))
    	{
		fscanf(elevation,"%lf%lf",&phi,&lam);
		i=round((phi-MinDatPhi)/PhiInt); 	j=round((lam-MinDatLam)/LamInt);
		fscanf(elevation,"%lf\n",&H(i,j));
    	}
    	fclose(elevation);
	if(density)
	{
		while(!feof(density))
    		{
			fscanf(density,"%lf%lf",&phi,&lam);
			i=round((phi-MinDatPhi)/PhiInt); 	j=round((lam-MinDatLam)/LamInt);
			fscanf(density,"%lf\n",&D(i,j));
    		}
    		fclose(density);
	}
// ***** PREPARATIONS TO COMPUTATION ******************************************************
	matris E(Nmax+2,M+2);			// Paul's coefficient
	matris Sn(M-1);				// Sn coefficients 
	matris bn(M+3);				// Bn coefficients
	n=2; m=0;
	mmax=(Mmax+1)*(Mmax+2)/2;
    	for(i=3;i<=mmax;i++)
    	{
		if(n==m)
		{
	    		c(n)=c(n)+pow(C(i),2)+pow(S(i),2);
	    		dc(n)=dc(n)+pow(cs(i),2)+pow(ss(i),2);
			c(n)=scale*(n-1.0)*(n-1.0)*c(n);
			dc(n)=scale*(n-1.0)*(n-1.0)*dc(n);
	    		m=0; n++;
		}
		else
		{
	    		m++;
	    		c(n)=c(n)+pow(C(i),2)+pow(S(i),2); 
	    		dc(n)=dc(n)+pow(cs(i),2)+pow(ss(i),2); 
		}
    	}
	for(n=Mmax+1;n<=Nmax;n++)		// Tcherning & Rapp (1974) Model
		c(n)=425.28*pow(0.999617,(n+2))*(n-1.0)/(n-2.0)/(n+24.0);
	t=cos(psi0);
	PN(0)=1.0;
	PN(1)=t;
	for(n=2;n<=Nmax+2;n++)			// Unnormalized Legendre function
		PN(n)=((2.0*n-1.0)/n)*t*PN(n-1)-((n-1.0)/n)*PN(n-2);
	E(0,0)=t+1.0;  	 
	E(1,1)=(t*t*t+1.0)/3.0; 
	E(2,0)=(6.0/5.0)*PN(0)*(PN(3)-PN(1))/6.0;	
	E(3,1)=((12.0/7.0)*PN(1)*(PN(4)-PN(2))-(2.0/3.0)*PN(3)*(PN(2)-PN(0)))/10.0; 
	E(2,2)=(9.0/10.0)*E(3,1)-(1.0/2.0)*E(2,0)+(3.0/5.0)*E(1,1); 
	E(4,2)=((PN(5)-PN(3))*PN(2)*20.0/9.0-(6.0/5.0)*PN(4)*(PN(3)-PN(1)))/14.0;
	E(3,3)=20.0/21.0*E(4,2)-2.0/3.0*E(3,1)+5.0/7.0*E(2,2);
	for(n=1;n<=M;n++)
 		E(n,0)=(PN(n+1)-PN(n-1))/(2.0*n+1.0);
	for(n=2;n<=M;n++)
		for(m=1;m<n;m++)
			E(n,m)=E(m,n)=((PN(n+1)-PN(n-1))*PN(m)*(n*(n+1.0))/(2.0*n+1.0)-((m*(m+1.0))/(2.0*m+1.0))*PN(n)*(PN(m+1)-PN(m-1)))/((n-m)*(n+m+1.0));
	for(n=M+1;n<=Nmax;n++)
		for(m=1;m<=M;m++)
		        E(n,m)=((PN(n+1)-PN(n-1))*PN(m)*(n*(n+1.0))/(2.0*n+1.0)-((m*(m+1.0))/(2.0*m+1.0))*PN(n)*(PN(m+1)-PN(m-1)))/((n-m)*(n+m+1.0));
 	for(n=2;n<=M;n++)
	 	E(n,n)=(((n+1.0)*(2.0*n-1.0))/(n*(2.0*n+1.0)))*E(n+1,n-1)-((n-1.0)/n)*E(n,n-2)+((2.0*n-1.0)/(2.0*n+1.0))*E(n-1,n-1);
	S_t=1.0/sin(psi0/2.0)-6.0*sin(psi0/2.0)+1.0-5.0*cos(psi0)-3.0*cos(psi0)*log(sin(psi0/2.0)+sin(psi0/2.0)*sin(psi0/2.0));
	tt=sqrt(1.0-t);
	dS_t=-8.0+3.0*sqrt(2.0)/tt+1.0/(sqrt(2.0)*tt*tt*tt)+3.0*(sqrt(2.0)-tt)/(sqrt(2.0)*(1.0-t*t))-3.0*log(0.5*tt*(sqrt(2.0)+tt)); 
	K0=-0.5+0.5*sqrt(2.0)/tt;
	K1=K0-1.0+tt/sqrt(2.0); 
	I2=(PN(3)-PN(1))/5.0;
	I(1)=(PN(2)-1.0)/3.0;
	K2=-1.0*K0+2.0*K1-I(1)/sqrt(2.0)/tt;
	for(n=2;n<=Nmax;n++)
	{
		I(n)=(PN(n+1)-PN(n-1))/(2.0*n+1.0);
		T(n)=(1.0/(2*n+1.0))*(n*I(n-1)+(n+1.0)*((1.0/(2.0*n+3.0))*(PN(n+2)-PN(n))));
		O(n)=K2;
		O(n+1)=2.0*K2-K1-I(n)/sqrt(2.0)/tt;
		K2=O(n+1);
		K1=O(n);
		Q(n)=-(n*S_t*(PN(n-1)-t*PN(n))-(1.0-t*t)*PN(n)*dS_t+2.0*O(n)+2.0*I(n)+9.0*T(n))/((n-1.0)*(n+2.0));
	}
// ** ESTIMATION OF MODIFICATION PARAMETERS ************************************************
	COEFFICIENT(M,v,C0,c,dc,Q,E,Sn,bn);
	t=sin(psi0/2.0);
    	Q0=-4.0*t+5.0*t*t+6.0*t*t*t-7.0*t*t*t*t+(6.0*t*t-6.0*t*t*t*t)*log(t)*(1.0+t);
	for(n=2;n<=M;n++)
		tot=tot+(2.0*n+1.0)/2.0*bn(n-2)*E(n,0);
	QM0=Q0-tot;
	mmax=(M+1)*(M+2)/2;
// ** COMPUTATION OF GRAVITY GRADYENTS ******************************************************
//	printf("%4i ",nthreads);
	pthread_t threads1[nthreads];
   	struct thread_data1 td1[nthreads];
   	int tdi;
   	double phistart;
	double phiend;
	for(tdi=0;tdi<nthreads;tdi++) 
	{	
  		phistart=floor((MinPhi-1+tdi*(MaxPhi-MinPhi+2)/nthreads)/PhiInt)*PhiInt+PhiInt/2;	
		phiend=ceil((MinPhi-1+(tdi+1)*(MaxPhi-MinPhi+2)/nthreads)/PhiInt)*PhiInt-PhiInt/2;
    		td1[tdi].thread_id=tdi;
		td1[tdi].phistart=phistart;
		td1[tdi].phiend=phiend;
		pthread_create(&threads1[tdi],NULL,thread1,(void *)&td1[tdi]);
   	}
	for(tdi=0;tdi<nthreads;tdi++) 
		pthread_join(threads1[tdi],nullptr);
// ** COMPUTATION OF GEOID HEIGHTS ***********************************************************
	pthread_t threads2[nthreads];
   	struct thread_data2 td2[nthreads];
	for(tdi=0;tdi<nthreads;tdi++) 
	{	
  		phistart=floor((MinPhi+tdi*(MaxPhi-MinPhi)/nthreads)/PhiInt)*PhiInt+PhiInt/2;	
		phiend=ceil((MinPhi+(tdi+1)*(MaxPhi-MinPhi)/nthreads)/PhiInt)*PhiInt-PhiInt/2;
		td2[tdi].thread_id=tdi;
		td2[tdi].phistart=phistart;
		td2[tdi].phiend=phiend;
		td2[tdi].QM0=QM0;
		td2[tdi].bn=&bn;
		pthread_create(&threads2[tdi],NULL,thread2,(void *)&td2[tdi]);
   	}
	for(tdi=0;tdi<nthreads;tdi++) 
		pthread_join(threads2[tdi],nullptr);
	if(s==0)
	{
		for(phi=MinPhi;phi<MaxPhi;phi=phi+PhiInt)
			for(lam=MinLam;lam<MaxLam;lam=lam+LamInt)
				printf("%12.8lf %12.8lf %8.4lf\n",phi,lam,geoid(round((phi-MinDatPhi)/PhiInt),round((lam-MinDatLam)/LamInt)));
	}
	else 
	{
		for(phi=MinPhi;phi<MaxPhi;phi=phi+PhiInt)
			for(lam=MinLam;lam<MaxLam;lam=lam+LamInt)
			{	
				in=round((phi-MinDatPhi)/PhiInt); 
				jn=round((lam-MinDatLam)/LamInt);	
				printf("%12.8lf %12.8lf %8.4lf %7.4lf %7.4lf %7.4f %7.4lf %8.4lf\n",phi,lam,approxN(in,jn),top(in,jn),atm(in,jn),dwc(in,jn),ell(in,jn),geoid(in,jn));
			}
	}
	clock_gettime(CLOCK_MONOTONIC, &finish);
	elapsed = (finish.tv_sec - start.tv_sec);
	elapsed += (finish.tv_nsec - start.tv_nsec)/1000000000.0;
//	printf("%10.4f \n",elapsed);
	return 0;
}// ** FINISH ***************************************************************************** 
void stokes_compute(double phistart, double phiend, double QM0, matris *bnn)
{
	int framePhi=round(psi0*rho/PhiInt);	// vertical limit of compartment
    	int frameLam=round(acos((cos(psi0)-pow(sin(MaxPhi/rho),2))/(pow(cos(MaxPhi/rho),2)))*rho/LamInt);// horizontal limit of compartment	
	int 	      m=0;
	int 	      n=0;
	int           i=0;			// array element
    	int           j=0;			// array element
	int           x=0;			// array element
    	int           y=0;			// array element
    	int          in=0;			// array element of computation point
    	int          jn=0;			// array element of computation point
	int	   imax=(Nmax+1)*(Nmax+2)/2;	// maximum array element for computation
	int        mmax=0;			// element of max expansion of GGM
	double       t=0.0;			// cos(psi)
	double     S_t=0.0;			// Stokes function (Closed form)
	double     phi=0.0;			// latitude of computational point [deg] 
    	double     lam=0.0;			// longitude of computational point [deg]
	double    phii=0.0;			// latitude of computational point [rad]
    	double    lami=0.0;			// longitude of computational point [rad]
	double     tot=0.0;			// total
	double    slat=0.0;			// spherical latitude of computational point [rad]
    	double    radi=0.0;			// spherical radius
 	double  stokes=0.0;			// stokes integration
	double   modif=0.0;			// modification part of Stokes' integral
	double   gamma=0.0;			// normal gravity on ellipsoid
	double    atm1=0.0;
	double      N2=0.0;			// N2 part of Stokes' integration
	double    DWC1=0.0;			// first part of DWC effect
	double    DWC2=0.0;			// second part of DWC effect
	double    DWC3=0.0;			// third part of DWC effect
	double  coslon=0.0;			// cos(mLamda)
    	double  sinlon=0.0;			// sin(mLamda)
    	double     Rr1=0.0;			// R/r
    	double     Rrn=0.0;			// R/r^n
    	double      Yn=0.0;			// Ymn
   	double    dpot=0.0;  			// disturbing potential [m**2/s**2]
    	double    gdst=0.0;  			// gravity disturbance [m/s**2]
	double   dpotn=0.0;  			// disturbing potential [m**2/s**2]
    	double   gdstn=0.0;  			// gravity disturbance [m/s**2]
	matris   SM(mx,mx);			// S^M(psi)*Blok Area		
	matris  l_0(mx,mx);			// spherical distance
	matris  psi(mx,mx);			// computational psi between P and Q
	matris  PN(Nmax+9);			// Unnormalized Legendre polynomial
	matris 	   P(imax);			// Normalized Legendre Function
	matris     Dgn(M+1); 			// Dg[n]
	matris    Dgnn(M+2); 			// Dgn[n]
	matris cosmlon(M+1);			// cos(mLongitute)
  	matris sinmlon(M+1);			// sin(mLongitute)
  	matris  &bn = *bnn;
	PN(0)=1.0;
	cosmlon(0)=1.0;
	mmax=(M+1)*(M+2)/2;
	for(phi=phistart;phi<phiend+PhiInt/2.0;phi+=PhiInt)
   	{
		lam=MinLam;
		in=round((phi-MinDatPhi)/PhiInt);	
		jn=round((lam-MinDatLam)/LamInt);
		for(i=in-framePhi;i<=in+framePhi;i++) 
		{
			x=i-in+framePhi; // position on the compartment
			for(j=jn;j<=jn+frameLam;j++)
			{
				if(i==in && j==jn) continue;
				y=abs(j-jn); // position on the compartment
				psi(x,y)=acos(sinlati(in)*sinlati(i)+coslati(in)*coslati(i)*cos(loni(j)-loni(jn)));
				if(psi(x,y)<=psi0)
				{
					t=cos(psi(x,y));
					tot=0.0;
					PN(1)=t;
					S_t=1.0/sin(psi(x,y)/2.0)-6.0*sin(psi(x,y)/2.0)+1.0-5.0*cos(psi(x,y))-3.0*cos(psi(x,y))*log(sin(psi(x,y)/2.0)+sin(psi(x,y)/2.0)*sin(psi(x,y)/2.0));
					for(n=2;n<=M;n++)
					{
						PN(n)=((2.0*n-1.0)/n)*t*PN(n-1)-((n-1.0)/n)*PN(n-2);
						tot=tot+(2.0*n+1.0)/2.0*bn(n-2)*PN(n);
					}
					SM(x,y)=(S_t-tot)*coslati(i);
				}
			}
		}
		phii=phi/rho;		
		slat=phii;
		radi=0.0;
		LevelEllipsoid levell(a,J2,GM,omega);		// Create normal gravity field (GRS80) 
		levell.ell2sph(&slat,&radi);
		LEGENDRE(P,M,sin(slat));		
		gamma=978032.67715*(1.0+0.0052790414*sin(phii)*sin(phii)+0.0000232718*pow(sin(phii),4)+0.0000001262*pow(sin(phii),6));
		for(lam=MinLam;lam<MaxLam;lam+=LamInt)
		{
			lami=lam/rho;
			in=round((phi-MinDatPhi)/PhiInt);		
			jn=round((lam-MinDatLam)/LamInt);
			for(i=in-framePhi;i<=in+framePhi;i++) 
			{
				for(j=jn-frameLam;j<=jn+frameLam;j++)
				{
					if(i==in && j==jn) j++;
					x=i-in+framePhi; // position on the compartment
					y=abs(j-jn); // position on the compartment
					if(psi(x,y)<=psi0)
					{
						stokes=stokes+SM(x,y)*(dg(i,j)-dg(in,jn));
						atm1+=SM(x,y)*H(in,jn);
						DWC3=DWC3+SM(x,y)*(H(in,jn)-H(i,j))*Dgr(i,j);
					}
				}
			}
			stokes=R/4.0/pi/gamma*constant*stokes;
			N2=R/2.0/gamma*dg(in,jn)*QM0;
			//  COMPUTATION OF Dg^GGM **************************************** /
			coslon=cos(lami);
			sinlon=sin(lami);
			cosmlon(1)=coslon;
			sinmlon(1)=sinlon;
			m=1;
			while(++m<=M)
			{
				cosmlon(m)=2.0*coslon*cosmlon(m-1)-cosmlon(m-2);
				sinmlon(m)=2.0*coslon*sinmlon(m-1)-sinmlon(m-2);
			}
			Rr1=a/radi;
			Rrn=1.0;
			n=2; m=0; i=3;
			while(i<mmax)
			{
				Yn +=P(i)*(C(i)*cosmlon(m)+S(i)*sinmlon(m));
				if(m==n)
				{
					Rrn  *= Rr1;
					dpot += Yn*Rrn;
        				gdst += Yn*Rrn*(n+1.0);
	  				dpotn = dpot*GM/a;
	       				gdstn = gdst*GM/a/radi;
	       				Dgn(n)= 1.0e+5*(gdstn-2.0*dpotn/radi);
					Dgnn(n)=Dgn(n)-Dgn(n-1);
					Yn=0.0;	n++; m=0;
				}
				else   m++; 
				i++;
			}
			for(n=2;n<=M;n++)
				modif=modif+bn(n-2)*Dgnn(n);
			modif=modif*R/2.0/gamma;
			approxN(in,jn)=modif-N2+stokes;
			// COMPUTATIONS OF ADDITIVE CORRECTIONS *******************************************
			if(density) top(in,jn)=-2.0e+5*pi*G*D(in,jn)/gamma*(pow(H(in,jn),2)+2.0*pow(H(in,jn),3)/3.0/R);
			else top(in,jn)=-2.0e+5*pi*G*2670.0/gamma*(pow(H(in,jn),2)+2.0*pow(H(in,jn),3)/3.0/R);
			atm(in,jn)=-1.0e+5*atm1*constant*G*1.23*R/gamma;
			ell(in,jn)=((0.0036-0.0109*sin(slat)*sin(slat))*dg(in,jn)+0.005*approxN(in,jn)*cos(slat)*cos(slat))*QM0;
			DWC1=H(in,jn)*(dg(in,jn)/gamma+3.0*approxN(in,jn)/(R+H(in,jn))-Dgr(in,jn)*H(in,jn)/2.0/gamma);
			for(n=2;n<=M;n++)
				DWC2=DWC2+bn(n-2)*(pow(R/(R+H(in,jn)),n+2)-1.0)*Dgnn(n);
			DWC2=DWC2*R/2.0/gamma;
			DWC3=DWC3*R/4.0/gamma/pi*constant;
			dwc(in,jn)=DWC1+DWC2+DWC3;
			geoid(in,jn)=approxN(in,jn)+top(in,jn)+atm(in,jn)+dwc(in,jn)+ell(in,jn);
			atm1=DWC2=DWC3=stokes=dpot=gdst=modif=0.0;
		}
	}
}
void gradient_compute(double phistart, double phiend)
{
	int   framePhi=round(0.5/PhiInt);	// vertical limit of compartment
	int   frameLam=round(acos((cos(0.5/rho)-pow(sin(MaxPhi/rho),2))/(pow(cos(MaxPhi/rho),2)))*rho/LamInt);//horizontal limit of compartment	
	int           i=0;			// array element
    	int           j=0;			// array element
	int           x=0;			// array element
    	int           y=0;			// array element
    	int          in=0;			// array element of computation point
    	int          jn=0;			// array element of computation point
	double     phi=0.0;			// latitude of computational point [deg] 
    	double     lam=0.0;			// longitude of computational point [deg]
	double    phii=0.0;			// latitude of computational point [rad]
    	double    lami=0.0;			// longitude of computational point [rad]
	double     tot=0.0;			// total 
	matris  l_0(mx,mx);			// spherical distance
	matris  psi(mx,mx);			// computational psi between P and Q
	for(phi=phistart;phi<phiend+PhiInt/2.0;phi+=PhiInt)
   	{
		lam=MinLam-1.0;
		in=round((phi-MinDatPhi)/PhiInt);		
		jn=round((lam-MinDatLam)/LamInt);
		for(i=in-framePhi;i<=in+framePhi;i++) 
		{
			x=i-in+framePhi; 	// position on the compartment
			for(j=jn;j<=jn+frameLam;j++)
			{
				if(i==in && j==jn) continue;
				y=abs(j-jn); 		// position on the compartment
				psi(x,y)=acos(sinlati(in)*sinlati(i)+coslati(in)*coslati(i)*cos(loni(j)-loni(jn)));
				if(psi(x,y)<0.008728) 	// capsize=0.5 degree for gravity gradient
					l_0(x,y)=pow(2.0*R*sin(psi(x,y)/2.0),3);
			}
		}
		for(lam=MinLam-1.0;lam<MaxLam+1.0;lam+=LamInt)
		{
			tot=0.0; 
			in=round((phi-MinDatPhi)/PhiInt);		
			jn=round((lam-MinDatLam)/LamInt);
			for(i=in-framePhi;i<=in+framePhi;i++) 
			{
				x=i-in+framePhi; 	// position on the compartment
				for(j=jn-frameLam;j<=jn+frameLam;j++)
				{
					if(i==in && j==jn) continue;
					y=abs(j-jn); 		// position on the compartment
					if(psi(x,y)<0.008728) 	// capsize=0.5 degree for gravity gradient
						tot=tot+(dg(i,j)-dg(in,jn))/l_0(x,y)*coslati(i);
				}
			}
			tot=tot*R*R/2.0/pi*constant;
			Dgr(in,jn)=tot-2.0*dg(in,jn)/R;
		}
	}
}
void COEFFICIENT(int M,int v,double C0,matris c,matris dc,matris Q,matris E,matris &Sn,matris &bn)
{
	int		n=0;			// degree
	int 		k=0;			// degree
	int 		r=0;			// order
	double  mu=0.99899012911838605;		// Constant for terrestrial data	
	double	cT=C0/(mu*mu);			// Constant for terrestrial data
	double     tot=0.0;			// total value
	matris  ST(Nmax+2);			// Sigma Terrestrial data 
	matris   q(Nmax+2);       		// qn coefficients
	matris      h(M-1);   		   	// observational vector  
	matris  A(M-1,M-1);			// Design matrix 
	matris  QM(Nmax+2); 		      	// Molodenski's Truncation Coefficient
	for(n=2;n<=Nmax;n++)
		ST(n)=cT*(1.0-mu)*pow(mu,n);
	if(v==1) // computation of coefficients Sn, bn for "Biased" version  
	{
		for(n=2;n<=Nmax;n++) 	
			q(n)=ST(n)+c(n);
		for(k=2;k<=M;k++) 
		{
			tot=0.0;
			for(n=2;n<=Nmax;n++) 
				tot=tot+(Q(n)*q(n)-2.0*ST(n)/(n-1.0))*E(n,k);
			h(k-2)=2.0*ST(k)/(k-1.0)-Q(k)*ST(k)+tot*(2.0*k+1.0)/2.0;
			for(r=k;r<=M;r++) 
			{
				tot=0.0;
				for(n=2;n<=Nmax;n++) 
					tot=tot+E(n,k)*E(n,r)*q(n);
				if(k==r) A(r-2,k-2)=ST(r)+dc(r)-(2.0*r+1.0)/2.0*E(k,r)*ST(r)-(2.0*k+1.0)/2.0*E(r,k)*ST(k)+(2.0*k+1.0)/2.0*(2.0*r+1.0)/2.0*tot;
				else     A(r-2,k-2)=A(k-2,r-2)=-(2.0*r+1.0)/2.0*E(k,r)*ST(r)-(2.0*k+1.0)/2.0*E(r,k)*ST(k)+(2.0*k+1.0)/2.0*(2.0*r+1.0)/2.0*tot;
			}
		}
		Sn=bn=invch(A)*h;
	}
	if(v==2) // computation of coefficients Sn, bn for "Unbiased" version 
	{
		for(n=2;n<=M;n++)      
			q(n)=ST(n)+dc(n);
		for(n=M+1;n<=Nmax;n++) 
			q(n)=ST(n)+c(n);
		for(k=2;k<=M;k++) 
		{
			tot=0.0;
			for(n=2;n<=Nmax;n++) 
				tot=tot+(Q(n)*q(n)-2.0*ST(n)/(n-1.0))*E(n,k);
			h(k-2)=2.0*ST(k)/(k-1.0)-Q(k)*q(k)+tot*(2.0*k+1.0)/2.0;
			for(r=k;r<=M;r++) 
			{
				tot=0.0; 
				for(n=2;n<=Nmax;n++) 
					tot=tot+E(n,k)*E(n,r)*q(n);
				if(k==r)    A(r-2,k-2)=q(k)-(2.0*r+1.0)/2.0*E(k,r)*q(r)-(2.0*k+1.0)/2.0*E(r,k)*q(k)+(2.0*k+1.0)/2.0*(2.0*r+1.0)/2.0*tot;
				else A(r-2,k-2)=A(k-2,r-2)=-(2.0*r+1.0)/2.0*E(k,r)*q(r)-(2.0*k+1.0)/2.0*E(r,k)*q(k)+(2.0*k+1.0)/2.0*(2.0*r+1.0)/2.0*tot;
			}
		}
    		Sn=solvsvd(A,h);
		for(n=2;n<=M;n++)
    		{
			tot=0.0;
	    		for(k=2;k<=M;k++)
				tot=tot+(2.0*k+1.0)/2.0*Sn(k-2)*E(n,k);
    			QM(n)=Q(n)-tot;
			bn(n-2)=Sn(n-2)+QM(n);
    		}
	}
	if(v==3) // computation of coefficients Sn, bn for "Optimum" version 
	{
		for(n=2;n<=M;n++)      
			q(n)=ST(n)+c(n)*dc(n)/(c(n)+dc(n));
		for(n=M+1;n<=Nmax;n++) 
			q(n)=ST(n)+c(n);
		for(k=2;k<=M;k++) 
		{
			tot=0.0;
			for(n=2;n<=Nmax;n++) 
				tot=tot+(Q(n)*q(n)-2.0*ST(n)/(n-1.0))*E(n,k);
			h(k-2)=2.0*ST(k)/(k-1.0)-Q(k)*q(k)+tot*(2.0*k+1.0)/2.0;
			for(r=k;r<=M;r++) 
			{
				tot=0.0;
				for(n=2;n<=Nmax;n++) 
					tot=tot+E(n,k)*E(n,r)*q(n);
				if(k==r) A(r-2,k-2)=q(k)-(2.0*r+1.0)/2.0*E(k,r)*q(r)-(2.0*k+1.0)/2.0*E(r,k)*q(k)+(2.0*k+1.0)/2.0*(2.0*r+1.0)/2.0*tot;
				else A(r-2,k-2)=A(k-2,r-2)=-(2.0*r+1.0)/2.0*E(k,r)*q(r)-(2.0*k+1.0)/2.0*E(r,k)*q(k)+(2.0*k+1.0)/2.0*(2.0*r+1.0)/2.0*tot;
			}
		}
    		Sn=solvsvd(A,h);
		for(n=2;n<=M;n++)
    		{
			tot=0.0;
	    		for(k=2;k<=M;k++)
				tot=tot+(2.0*k+1.0)/2.0*Sn(k-2)*E(n,k);
    			QM(n)=Q(n)-tot;
			bn(n-2)=(Sn(n-2)+QM(n))*c(n)/(c(n)+dc(n));
    		}
	}
}
void LEGENDRE(matris &P,int N,double x)
{
	int i=0;
	int j=0;
	int k=0;
	int n=0;
	int m=0;
	int imax=(N+1)*(N+2)/2;
	double sinx=x; 
	double cosx=sqrt(1-x*x);
	double f1=0.0;
	double f2=0.0;
	double f3=0.0;
	double f4=0.0;
	double f5=0.0;
	P(0)=1.0;
	P(1)=sqrt(3.0)*sinx;
	P(2)=sqrt(3.0)*cosx;
	for(n=2;n<=N;n++)
	{
		i=n*(n+1)/2+n;				// index for Pn,n
		j=i-n-1;				// index for Pn-1,n-1
		f1=sqrt((2.0*n+1)/2/n);
		f2=sqrt(2.0*n+1);
		P(i)=f1*cosx*P(j);			// diagonal elements
		P(i-1)=f2*sinx*P(j);			// subdiagonal elements
	}
	n=2;	m=0;	i=2;
	while(++i<imax-2)
	{
		if(m==n-1)
		{
			m=0; n++; i++;
		}
		else
		{
			j=i-n;						// index for Pn-1,m
			k=i-2*n+1;					// index for Pn-2,m
			f3=sqrt((2.0*n+1)/(n-m)/(n+m));
			f4=sqrt(2.0*n-1);
			f5=sqrt((n-m-1.0)*(n+m-1.0)/(2.0*n-3));
			P(i)=f3*(f4*sinx*P(i-n)-f5*P(k));
			m++;
		}
	}
}
void HELP()
{
	fprintf(stderr,"\n     The LSMSSOFT2.0 computes a gravimetric geoid model by the KTH method and additive corrections.\n\n");
	fprintf(stderr,"USAGE:\n");
	fprintf(stderr,"     LSMSSOFT2.0 -G[model<file>] -A[anomaly<file>] -E[elevation<file>] -D[density<file>] -M[<value>]\n");
	fprintf(stderr,"             ... -P[<value>] -C[<value>] -V[<value>] -R[<value>] -I[<value>] -S -H\n\n");
	fprintf(stderr,"PARAMETERS:\n");
	fprintf(stderr,"     model:     global geopotential model that will be used as a reference model.\n");
	fprintf(stderr,"                It includes harmonic coefficients in GFZ format (n,m,Cnm,Snm,sigmaC,sigmaS).\n\n");
	fprintf(stderr,"     anomaly:   mean free-air gravity anomalies which cover the data area.\n");
	fprintf(stderr,"                It includes grid based data (latitude, longitude, and mean anomaly).\n\n");
	fprintf(stderr,"     elevation: mean topographic elevations which cover the data area.\n");
	fprintf(stderr,"                It includes grid based DEM data (latitude, longitude, and mean elevation).\n\n");
	fprintf(stderr,"OPTIONS:\n");
	fprintf(stderr,"     density:   mean crust density which cover the target area (if desired).\n");
	fprintf(stderr,"                It includes grid based density data (latitude, longitude, and mean density).\n\n");
	fprintf(stderr,"     -M<value>  maximum expansion of the GGM used in the computation\n");
	fprintf(stderr,"                default: 115\n\n");
	fprintf(stderr,"     -P<value>  integration capsize (unit: degree)\n");
	fprintf(stderr,"                default: 1.0 \n\n");
	fprintf(stderr,"     -C<value>  variance of terrestrial gravity data (unit: mGal^2)\n");
	fprintf(stderr,"                default: 16 \n\n");
	fprintf(stderr,"     -V<value>  version of the KTH method (biased=1, unbiased=2, optimum=3)\n");
	fprintf(stderr,"                default: 1\n\n");
	fprintf(stderr,"     -R<value>  limits of the target area (MinLat/MaxLat/MinLon/MaxLon)\n");
	fprintf(stderr,"                default: 45.01/46.99/2.01/3.99\n\n");
	fprintf(stderr,"     -I<value>  intervals of the grid (LatInterval/LonInterval)\n");
	fprintf(stderr,"                default: 0.02/0.02 (72x72 arc-seconds)\n\n");
	fprintf(stderr,"     -S         prints all segments (lat, lon, Ñ, top, atm, dwc, ell, N), respectively.\n\n");
	fprintf(stderr,"     -H         prints this help\n\n");
	fprintf(stderr,"AUTHOR:         Dr. R. Alpay ABBAK\n\n");
        exit(EXIT_SUCCESS);
}
