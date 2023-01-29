#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
using namespace std;
using namespace cv;

using namespace cv;
const int max_value_H = 360 / 2;
const int max_value = 255;
const String window_capture_name = "Video Capture";
const String window_detection_name = "Object Detection";
int low_H = 0, low_S = 0, low_V = 0;
int high_H = max_value_H, high_S = max_value, high_V = max_value;

Mat imagen = imread("prueba.png");
Mat mascara, ImagenLap, borde, binarizado, distancia;
Mat Laplacianmas = (Mat_<float>(3, 3) << 1, 1, 1, 1, -8, 1, 1, 1, 1);
vector<Vec3b> colores;

#pragma region TonoDePiel
static void on_low_H_thresh_trackbar(int, void*)
{
    low_H = min(high_H - 1, low_H);
    setTrackbarPos("Low H", window_detection_name, low_H);
}
static void on_high_H_thresh_trackbar(int, void*)
{
    high_H = max(high_H, low_H + 1);
    setTrackbarPos("High H", window_detection_name, high_H);
}
static void on_low_S_thresh_trackbar(int, void*)
{
    low_S = min(high_S - 1, low_S);
    setTrackbarPos("Low S", window_detection_name, low_S);
}
static void on_high_S_thresh_trackbar(int, void*)
{
    high_S = max(high_S, low_S + 1);
    setTrackbarPos("High S", window_detection_name, high_S);
}
static void on_low_V_thresh_trackbar(int, void*)
{
    low_V = min(high_V - 1, low_V);
    setTrackbarPos("Low V", window_detection_name, low_V);
}
static void on_high_V_thresh_trackbar(int, void*)
{
    high_V = max(high_V, low_V + 1);
    setTrackbarPos("High V", window_detection_name, high_V);
}

void TonoDePiel() {    
    namedWindow(window_capture_name);
    namedWindow(window_detection_name);
    createTrackbar("Low H", window_detection_name, &low_H, max_value_H, on_low_H_thresh_trackbar);
    createTrackbar("High H", window_detection_name, &high_H, max_value_H, on_high_H_thresh_trackbar);
    createTrackbar("Low S", window_detection_name, &low_S, max_value, on_low_S_thresh_trackbar);
    createTrackbar("High S", window_detection_name, &high_S, max_value, on_high_S_thresh_trackbar);
    createTrackbar("Low V", window_detection_name, &low_V, max_value, on_low_V_thresh_trackbar);
    createTrackbar("High V", window_detection_name, &high_V, max_value, on_high_V_thresh_trackbar);
    Mat frame = imread("rostro.jpg");
     Mat frame_HSV, frame_threshold;
    while (true) {
        // Convert from BGR to HSV colorspace
        cvtColor(frame, frame_HSV, COLOR_BGR2HSV);
        // Detect the object based on HSV Range Values
        inRange(frame_HSV, Scalar(low_H, low_S, low_V), Scalar(high_H, high_S, high_V), frame_threshold);
        // Show the frames
        imshow(window_capture_name, frame);
        imshow(window_detection_name, frame_threshold);
        char key = (char)waitKey(30);
        if (key == 'q' || key == 27)
        {
            break;
        }
    }
}
#pragma endregion

#pragma region SegmentacionPorColor
void SegmentacionColor() {
    imshow("Imagen Original", imagen);
    /*En caso de que el fondo sea transparente o blanco etc, se forzara a ser oscuro
    así es más facil la detección de objetos*/

    inRange(imagen, Scalar(255, 255, 255), Scalar(255, 255, 255), mascara);
    imagen.setTo(Scalar(0, 0, 0), mascara);

    /*Se usan valores de 32 bits debido a que esta mascara puede generar algunos valores negativos
    y pueden perderse cosa que no es muy buena*/
    filter2D(imagen, ImagenLap, CV_32F, Laplacianmas);
    imagen.convertTo(borde, CV_32F);
    Mat imgResul = borde - ImagenLap;
    //Convertimos la imagen a grises
    imgResul.convertTo(imgResul, CV_8UC3);
    ImagenLap.convertTo(ImagenLap, CV_8UC3);
    imshow("Contorno de la imagen", imgResul);

    //Imagen binarizada
    cvtColor(imgResul, binarizado, COLOR_BGR2GRAY);
    threshold(binarizado, binarizado, 40, 255, THRESH_BINARY | THRESH_OTSU);
    //Algoritmo para la distancia
    distanceTransform(binarizado, distancia, DIST_L2, 3);
    //Se normaliza un rango determinado apra la binarización en este caso de 0 a 1
    normalize(distancia, distancia, 0, 1.0, NORM_MINMAX);
    threshold(distancia, distancia, 0.4, 1.0, THRESH_BINARY);
    Mat kernel1 = Mat::ones(3, 3, CV_8U);
    //Algoritmo de dilatación
    dilate(distancia, distancia, kernel1);

    //Version de la imagen para hallar los contornos, se necesita gris de unsigned 8 bits
    Mat dist_8u;
    distancia.convertTo(dist_8u, CV_8U);
    // Find total semillas
    vector<vector<Point>> contornos;
    findContours(dist_8u, contornos, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    //Hacemos los marcadores para el watershed
    Mat semillas = Mat::zeros(distancia.size(), CV_32S);
    // Draw the foreground semillas
    for (size_t i = 0; i < contornos.size(); i++)
    {
        drawContours(semillas, contornos, static_cast<int>(i), Scalar(static_cast<int>(i) + 1), -1);
    }

    //Semillas
    circle(semillas, Point(5, 5), 3, Scalar(255), -1);
    Mat semillas8u;
    semillas.convertTo(semillas8u, CV_8U, 10);

    //Ejecutamos el algoritmo watershed
    watershed(imgResul, semillas);
    Mat semilla;
    semillas.convertTo(semilla, CV_8U);

    /*Validación de la imagen binarizada
    1 valida 0 / 0 valida 1*/
    bitwise_not(semilla, semilla);
    //Marcadores de la imagen
    imshow("Semillas Watershed", semilla);

    for (size_t i = 0; i < contornos.size(); i++)
    {
        int b = theRNG().uniform(0, 256);
        int g = theRNG().uniform(0, 256);
        int r = theRNG().uniform(0, 256);
        colores.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
    }

    Mat imgResultado = Mat::zeros(semillas.size(), CV_8UC3);

    // Colores random para cada contorno
    for (int i = 0; i < semillas.rows; i++)
    {
        for (int j = 0; j < semillas.cols; j++)
        {
            int index = semillas.at<int>(i, j);
            if (index > 0 && index <= static_cast<int>(contornos.size()))
            {
                imgResultado.at<Vec3b>(i, j) = colores[index - 1];
            }
        }
    }
    imshow("Contornos", imgResultado);
    waitKey(0);
}
#pragma endregion


#pragma region Averiguacion
void Segmentacion() {
    //Dividir y fusionar
    Mat image = imread("rgb.png");
    // Mat image = imread("water_coins.png");
    imshow("Imagen Original", image);
    std::vector<Mat> channels;
    split(image, channels);

    Mat zero = Mat::zeros(image.size(), CV_8UC1);

    std::vector<Mat> B = { channels[0], zero, zero };
    std::vector<Mat> G = { zero, channels[1], zero };
    std::vector<Mat> R = { zero, zero, channels[2] };

    Mat rdst, gdst, bdst;

    merge(R, rdst);
    merge(G, gdst);
    merge(B, bdst);

    imshow("R Channel", rdst);
    imshow("G Channel", gdst);
    imshow("B Channel", bdst);

    waitKey(0);
}
#pragma endregion

int main()
{
    int op;
    cin >> op;
    switch(op){
    case 1:
        SegmentacionColor();
        break;
    case 2:
        TonoDePiel();
        break;
    case 3:
        break;
    case 4:
        Segmentacion();
        break;
    }
    return 0;
}