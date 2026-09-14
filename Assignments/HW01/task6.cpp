#include <iostream>

int main(int argc, char *argv[]) {
    int N = std::atoi(argv[1]);

    for (int i = 0; i <= N; i++)
        std::printf("%d ", i);

    std::printf("\n");

    for (int i = N; i >=0; i--)
        std::cout << i << " ";
    
    std::cout << std::endl;
}