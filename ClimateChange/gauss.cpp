#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include "Vector.hpp"
#include "RandomGen.hpp"



std::vector<double> gaussian_random_walk_ff(double start_point, int num_points, double a, double mean, double sigma,RandomGen gen, int salto){
    
    std::vector<double> walk(num_points + 1);
    walk[0] = start_point;

    for (int i = 1; i <= num_points; ++i) {


        double increment = gen.Gauss(mean, sigma);
        //double bump_a = ((i-1) % 2 == 0) ? std::abs(a) : -std::abs(a); // Oscilla tra positivo e negativo ad ogni passo        
        double bump_a = ((i % salto) == 0) ? -std::abs(a) : std::abs(a);
        // Calcola le probabilità p e q basate sul punto corrente
        double current_point = walk[i-1];
        double p = 0.5*(1+std::erf((bump_a*current_point-mean)/(sigma*sqrt(2))));
        double q = 1 - p;

        // Decide se modificare l'incremento in base a p e q
        if (p > q) {
            increment = -std::abs(increment);
        } else if (p < q) {
            increment = std::abs(increment);
        }
        // Se p = q, non modifica l'incremento

        walk[i] = walk[i-1] + increment;

    }

    return walk;
}

std::vector<double> read_means(){
    std::vector<double> means;
    std::ifstream infile("means.txt");
    std::string line;

    while (std::getline(infile, line)) {
        means.push_back(std::stod(line));  // Converti la stringa in double e aggiungila a "means"
    }
    infile.close();
    return means;
}

std::vector<double> gaussian_random_walk_sigmas(double start_point, int num_points, double a, double mean, double sigma, RandomGen& gen, int salto) {
    std::vector<double> walk(num_points + 1);
    walk[0] = start_point;
    double currentSum = start_point;

    for (int i = 1; i <= num_points; ++i) {
        //double bump_a = ((i-1) % 2 == 0) ? std::abs(a) : -std::abs(a);
        double p = 0.5 * (1 + std::erf((a * currentSum - mean) / (sigma * sqrt(2))));
        double q = 1 - p;
        double increment = gen.ConstrainedGauss(mean, sigma, currentSum, i);

        if (p > q) {
            increment = -std::abs(increment);
        } else if (p < q) {
            increment = std::abs(increment);
        }

        walk[i] = walk[i-1] + increment;
        currentSum += increment;
    }

    return walk;
}

// std::vector<double> generate_walks(const std::vector<double>& means, int points_per_mean, int a, double sigma, RandomGen gen) {
//     std::vector<double> all_walks;

//     for (int i = 0; i < means.size(); ++i) {

//         double mean = means[i];
//         std::vector<double> walk;
//         // Supponiamo che tu voglia applicare il tuo "tool" solo ai valori di "mean"
//         // che soddisfano una certa condizione. Qui, come esempio, utilizziamo una
//         // condizione semplice: applichiamo il tool solo se "mean" è maggiore di 10.
//         if (i==31 || i == 44) {
//             walk = gaussian_random_walk_feedback(mean, points_per_mean - 1, a, mean, sigma, gen);
//             all_walks.insert(all_walks.end(), walk.begin(), walk.end());
//         } else {
//             walk = gaussian_random_walk_feedback(mean, points_per_mean - 1, a, mean, sigma, gen);
//             all_walks.insert(all_walks.end(), walk.begin(), walk.end());        
//         }
//     }

//     return all_walks;
// }

int main() {   

    double ef_param[] {0.124751, 0.2657952800487994, -1.3322676295501879e-18,  1499,};
    double df_param[] {  6.661338e-16, 0.23806519451569588, -3.8607110340190925e-17, 774 };
    double gf_param[] {0.053458, 0.09155913168201571, -4.9167019661971215e-18, 174};

    double start_point = gf_param[0];
    double sigma =gf_param[1];
    double mean =gf_param[2];
    int num_points =gf_param[3]; // Numero di punti in ciascuna simulazione

    int num_simulations_1 = 10000; // Numero di simulazioni da eseguire
    int num_simulations_2 = 1000; // Numero di simulazioni da eseguire
    
    int salto = 17;
    cout << "\nSalto tra i passi: " << salto << endl;
    std::ofstream outfile("simulations.txt", std::ios::out); // Apre in modalità scrittura, sovrascrivendo file esistenti
    std::ofstream outfile2("simu-blocks.txt", std::ios::out);
    std::vector<double> goog;
    for (int sim = 0; sim < num_simulations_1; ++sim) {
        double a = 1; 
        RandomGen myGen(sim+1);
        double increment = gen.Gauss(mean, sigma);
        
        //std::vector<double> walk = gaussian_random_walk_ff(start_point, num_points, a, mean, sigma, myGen, salto);
        if (sim==0 || sim==1){
            cout << walk[33] << "\n";
        }
        for (double point : walk) {
            outfile << point << " ";
        }
        outfile << "\n"; // Nuova simulazione su nuova linea
    }
/*/
    for (int sim = 0; sim < num_simulations_2; ++sim) {
        double a = 1;
        std::vector<double> means = read_means();
        RandomGen myGen2(sim+1);
        std::vector<double> walk2 = generate_walks(means, 30, a , sigma, myGen2); // valori da cambiare!!!!!!
        for (double point : walk2) {
            outfile2 << point << " ";
        }
        outfile2 << "\n"; // Nuova simulazione su nuova linea
    }    
/*/
    outfile.close();
    outfile2.close();

    std::cout << "Simulazioni completate e salvate in simulations.txt" << std::endl;

    return 0;
}

