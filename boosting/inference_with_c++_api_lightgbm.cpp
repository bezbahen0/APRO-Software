#include <LightGBM/application.h>
#include <iostream>

int main(int argc, char **argv)
{
    LightGBM::Application app = LightGBM::Application(argc, argv);
    app.Run();
    return EXIT_SUCCESS;
}
