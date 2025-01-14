#include <fstream>
#include <iostream>
#include <random>
#include "Vector.hpp"
#include "RandomGen.hpp"

int main() {
    const int num_simulations = 10000;
    const int num_points = 1500;

    std::ofstream out_file("data.txt");
    std::default_random_engine generator;


    for (int i = 0; i < num_simulations; ++i) {
        RandomGen gen(i+1);
        for (int j = 0; j < num_points; ++j) {
            double number = gen.Gauss(0,0.2657952800487994);
            out_file << number;
            if (j < num_points - 1) out_file << " ";
        }
        out_file << "\n";
    }

    out_file.close();
    std::cout << "File scritto con successo." << std::endl;

    return 0;
}
