# LSMSSOFT2.0
/* The LSMSSOFT2.0 computes a gravimetric geoid by using the KTH method and additive corrections.
 * 
 * This program is distributed under the terms of a GNU Public License (ver. 1).
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
