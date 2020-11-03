#include "opencv2/opencv.hpp"
#include "opencv2/dnn.hpp"
#include "track.hpp"

int main(int argc, char **argv)
{
	TRACK tracker(10);
	String modelTxt = "model/deploy.prototxt";
	String modelBin = "model/res10_300x300_ssd_iter_140000.caffemodel";
	// String vidPath = "mydata/videos/cam21_20201023.avi";

	auto net = dnn::readNetFromCaffe(modelTxt, modelBin);
	float confidenceThreshold = 0.2;

	VideoCapture cap(0);
	int delay = 1;
	int frame_id = 0;

	while (1)
	{
		Mat frame;
		cap >> frame;
		if (frame.empty())
			break;

		// Mat orig_frame = frame.clone();
		resize(frame, frame, Size(400, 300));
		auto inputBlob = dnn::blobFromImage(frame, 1.0, Size(400, 300), Scalar(104.0, 177.0, 123.0));
		net.setInput(inputBlob);
		auto detection = net.forward("detection_out");
		Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

		vector<struct Bbox> bboxes;
		vector<BoundingBox> boxes;
		vector<TrackingBox> detFrameData;

		for (int i = 0; i < detectionMat.rows; i++)
		{
			float confidence = detectionMat.at<float>(i, 2);

			if (confidence > confidenceThreshold)
			{
				int xLeftTop = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
				int yLeftTop = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
				int xRightBottom = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
				int yRightBottom = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);

				struct Bbox bbox;
				bbox.score = confidence;
				bbox.x1 = xLeftTop;
				bbox.y1 = yLeftTop;
				bbox.w = xRightBottom - xLeftTop;
				bbox.h = yRightBottom - yLeftTop;
				bboxes.push_back(bbox);
			}
		}

		for (vector<struct Bbox>::iterator it = bboxes.begin(); it != bboxes.end(); it++)
		{
			boxes.push_back(BoundingBox(*it));
		}

		for (int i = 0; i < boxes.size(); ++i)
		{
			TrackingBox cur_box;
			cur_box.box = boxes[i].rect;
			cur_box.id = i;
			cur_box.frame = frame_id;
			detFrameData.push_back(cur_box);
		}
		++frame_id;

		vector<TrackingBox> tracking_results = tracker.update(detFrameData);

		for (TrackingBox it : tracking_results)
		{
			Rect object(it.box.x, it.box.y, it.box.width, it.box.height);
			rectangle(frame, object, tracker.randColor[it.id % 255], 2);
			putText(frame,
					to_string(it.id),
					Point2f(it.box.x, it.box.y),
					FONT_HERSHEY_PLAIN,
					2,
					tracker.randColor[it.id % 255]);
		}

		imshow("Webcam", frame);
		if ((waitKey(delay) == 113))
			break;
	}

	cap.release();
	destroyAllWindows();

	return 0;
}