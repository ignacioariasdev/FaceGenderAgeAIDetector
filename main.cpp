#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

using namespace cv;
using namespace dnn;
using namespace std;

// Age categories
const vector<string> ageList = {"(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"};
// Gender categories
const vector<string> genderList = {"Male", "Female"};

Mat preprocess(Mat &frame, Size targetSize) {
    Mat blob = blobFromImage(frame, 1.0, targetSize, Scalar(104.0, 177.0, 123.0), false, false);
    return blob;
}

int main() {
    cout << "Current path is " << fs::current_path() << endl;

    // Paths to models
    String modelDir = "../Models/";  // Adjust this path if needed
    String faceModel = modelDir + "res10_300x300_ssd_iter_140000.caffemodel";
    String faceProto = modelDir + "face_deploy.prototxt";
    String ageModel = modelDir + "age_net.caffemodel";
    String ageProto = modelDir + "age_deploy.prototxt";
    String genderModel = modelDir + "gender_net.caffemodel";
    String genderProto = modelDir + "gender_deploy.prototxt";

    // Check if files exist
    vector<String> files = {faceModel, faceProto, ageModel, ageProto, genderModel, genderProto};
    for (const auto& file : files) {
        if (!fs::exists(file)) {
            cerr << "Error: File not found: " << file << endl;
            return -1;
        }
    }

    // Load face detection model
    Net faceNet = readNet(faceProto, faceModel);
    // Load age and gender detection models
    Net ageNet = readNet(ageProto, ageModel);
    Net genderNet = readNet(genderProto, genderModel);

    // Open default camera (0 = built-in camera)
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error: Could not open camera!" << endl;
        return -1;
    }

    while (true) {
        Mat frame;
        cap >> frame;  // Capture a new frame
        if (frame.empty()) {
            cerr << "Error: Captured frame is empty!" << endl;
            break;
        }

        // Convert to blob for face detection
        Mat blob = preprocess(frame, Size(300, 300));
        faceNet.setInput(blob);
        Mat detection = faceNet.forward();

        // Loop over all detected faces
        detection = detection.reshape(1, detection.total() / 7);
        for (int i = 0; i < detection.rows; ++i) {
            float confidence = detection.at<float>(i, 2);
            if (confidence > 0.7) {  // Confidence threshold
                // Extract face box
                int x1 = max(0, static_cast<int>(detection.at<float>(i, 3) * frame.cols));
                int y1 = max(0, static_cast<int>(detection.at<float>(i, 4) * frame.rows));
                int x2 = min(frame.cols, static_cast<int>(detection.at<float>(i, 5) * frame.cols));
                int y2 = min(frame.rows, static_cast<int>(detection.at<float>(i, 6) * frame.rows));

                // Ensure the face box has positive width and height
                if (x2 <= x1 || y2 <= y1) {
                    continue;  // Skip this detection
                }

                Rect faceBox(x1, y1, x2 - x1, y2 - y1);

                // Draw the face box
                rectangle(frame, faceBox, Scalar(0, 255, 0), 2);

                // Extract the face region
                Mat face = frame(faceBox);

                try {
                    // Predict gender
                    Mat genderBlob = preprocess(face, Size(227, 227));
                    genderNet.setInput(genderBlob);
                    Mat genderPreds = genderNet.forward();
                    int genderIdx = genderPreds.at<float>(0) > 0.5 ? 1 : 0;  // Male/Female
                    string gender = genderList[genderIdx];

                    // Predict age
                    Mat ageBlob = preprocess(face, Size(227, 227));
                    ageNet.setInput(ageBlob);
                    Mat agePreds = ageNet.forward();
                    int ageIdx = max_element(agePreds.begin<float>(), agePreds.end<float>()) - agePreds.begin<float>();
                    string age = ageList[ageIdx];

                    // Display age and gender label
                    string label = gender + ", " + age;
                    putText(frame, label, Point(x1, y1 - 10), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 255), 2);
                } catch (const cv::Exception& e) {
                    cerr << "OpenCV error: " << e.what() << endl;
                    // Continue with the next detection
                }
            }
        }

        // Show the frame with detections
        imshow("Face, Age, and Gender Detection", frame);

        // Break the loop if 'q' is pressed
        if (waitKey(1) == 'q') {
            break;
        }
    }

    return 0;
}