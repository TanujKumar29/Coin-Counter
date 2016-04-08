package app.tanuj.com.detectandcountcoins;
import android.Manifest;
import android.annotation.TargetApi;
import android.app.Activity;
import android.content.Context;
import android.content.ContextWrapper;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.RectF;
import android.media.Image;
import android.net.Uri;
import android.nfc.Tag;
import android.os.Environment;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.ActionBarActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;
import org.opencv.android.OpenCVLoader;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

import static org.opencv.core.CvType.*;


public class MainActivity extends Activity {
    Mat circles;
    public static final int VIEW_MODE_RGBA = 0;
    public static int viewMode = VIEW_MODE_RGBA;
    private static final int CAMERA_REQUEST = 1888;
    private ImageView imageView;
    public String imgPath=Environment.getExternalStorageDirectory().getAbsolutePath();
    //private BaseLoaderCallback mLoaderCallback;
    Bitmap photo;
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i("base loader call back", "OpenCV loaded successfully");
                    // mOpenCvCameraView.enableView();
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };
    public MainActivity() {
        Log.i("main activity log", "Instantiated new " + ((Object) this).getClass());
    }
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        this.imageView = (ImageView)this.findViewById(R.id.imageBox_image_view);
        Button photoButton = (Button) this.findViewById(R.id.TakeImage_button);
        photoButton.setOnClickListener(new View.OnClickListener() {

            @Override
            public void onClick(View v) {
                Intent cameraIntent = new Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE);
                startActivityForResult(cameraIntent, CAMERA_REQUEST);
            }
        });

        boolean hasPermission = (ContextCompat.checkSelfPermission(this,
                Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED);
        if (!hasPermission) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE},
                    112);
        }
    }
    @Override
    public void onResume() {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_3, this,
                mLoaderCallback);
    }


    public void display(Bitmap photo)
    {
        circles=new Mat();
        Mat grad_x = new Mat();
        Mat grad_y = new Mat();
        Mat abs_grad_x = new Mat();
        Mat abs_grad_y = new Mat();
        Mat grad = new Mat();
        int skala = 1;
        int delta = 0;
        int depth = 3; //CV16_S
        int scale=1;
        List<Mat> hull = new ArrayList();
        Mat drawing;
        double M_PI =3.14159265358979323846;
        Mat src=new Mat();
        Mat srcGray=new Mat();
        Mat srcGaussianBlur=new Mat();

        Mat srcCanny=new Mat();
        Matrix m = new Matrix();
        m.setRectToRect(new RectF(0, 0, photo.getWidth(), photo.getHeight()), new RectF(0, 0, 1000, 1000), Matrix.ScaleToFit.CENTER);
        Bitmap bbit = Bitmap.createBitmap(photo, 0, 0,photo.getWidth(), photo.getHeight(), m, true);

        //  name = Environment.getExternalStorageDirectory().getAbsolutePath();
        Toast.makeText(getApplicationContext(), ""+imgPath, Toast.LENGTH_SHORT).show();
        try
        {
            OutputStream stream = new FileOutputStream(imgPath+"/image.jpg");
            bbit.compress(Bitmap.CompressFormat.PNG, 100, stream);
        }
        catch (Exception ex)
        {
            Toast.makeText(getApplicationContext(),"EXCEPTION "+ex.getMessage(), Toast.LENGTH_LONG).show();
        }

        Utils.bitmapToMat(bbit, src, true);
        Highgui.imwrite(imgPath + "/Mat.jpg", src);

        Mat processingFrame=new Mat();
        Imgproc.cvtColor(src, processingFrame, Imgproc.COLOR_RGBA2BGR, 3);
        Imgproc.GaussianBlur(processingFrame, processingFrame, new org.opencv.core.Size(3,3),2, 2, Imgproc.BORDER_DEFAULT);
        Imgproc.cvtColor(processingFrame, srcGray, Imgproc.COLOR_RGBA2GRAY);
        Imgproc.Sobel(srcGray, grad_x, CvType.CV_16S, 1, 0, 3, scale, delta,Imgproc.BORDER_DEFAULT);
        Imgproc.Sobel(srcGray, grad_x, CvType.CV_16S, 1, 0, 3, scale, delta);
        Core.convertScaleAbs(grad_x, abs_grad_x); // / Gradient Y //
        Imgproc.Sobel(srcGray, grad_y, CvType.CV_16S, 0, 1, 3, scale, delta,
                Imgproc.BORDER_DEFAULT);
        Imgproc.Sobel(srcGray, grad_y, CvType.CV_16S, 0, 1, 3, scale, delta);
        Core.convertScaleAbs(grad_y, abs_grad_y);
        // / Total Gradient(approximate)
        Core.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
        Highgui.imwrite(imgPath + "/grad.jpg", grad);

       // Imgproc.bilateralFilter(grad, srcGray, 100,1,Imgproc.BORDER_DEFAULT);
      //  Imgproc.medianBlur(grad, grad, 5);
      //  Highgui.imwrite(imgPath + "/bilateralFilter.jpg", srcGray);
       // Log.i("gray","gray");

       /* Imgproc.GaussianBlur(grad, grad, new Size(1,1),2,2);
        Highgui.imwrite(imgPath + "/GaussianBlur1.jpg", grad);
*/
        Imgproc.GaussianBlur(grad, grad, new org.opencv.core.Size(3,3),2, 2, Imgproc.BORDER_DEFAULT);
        Imgproc.Canny(grad, grad, 10,30);
        Highgui.imwrite(imgPath + "/Canny.jpg", grad);
        Log.i("canny","canny");

       /* Imgproc.adaptiveThreshold(grad, grad, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY,15,8);
        Highgui.imwrite(imgPath + "/threshAdapt.jpg", grad);*/

       Imgproc.HoughCircles(grad, circles, Imgproc.CV_HOUGH_GRADIENT,1,grad.rows()/8,100,15,20,60);
       // Toast.makeText(getApplicationContext(), ""+circles.cols()+"rows"+grad.rows(), Toast.LENGTH_SHORT).show();
        /// Draw the circles detected
        // int size=(int)circles.size();
        for( int i = 0; i<= circles.cols(); i++)
        {
            double vCircle[] = circles.get(0,i);
            if (vCircle == null)
                break;
            Point pt = new Point(Math.round(vCircle[0]),Math.round(vCircle[1]));
            int radius = (int) Math.round(vCircle[2]);
            Log.d("cv", pt + " radius " + radius);
           // Core.circle(grad,pt,3,new Scalar(0,0,0), -1, 8, 0);

            Core.circle(grad,pt,radius,new Scalar(0,0,0), 3, 8, 0 );
            Highgui.imwrite(imgPath + "/coreCircle.jpg", grad);
        }


    }
    public void displayFinalVal(View view) {
        TextView quantityTextView = (TextView) findViewById(
                R.id.DisplayVal_text_view);
        quantityTextView.setText("Number of coins =" + circles.cols());

    }


    @TargetApi(23)
    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        //switch (requestCode)
        {
            {
                if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED)
                {
                    //reload my activity with permission granted or use the features what required the permission
                } else
                {
                    Toast.makeText(getApplicationContext(), "The app was not allowed to write to your storage. Hence, it cannot function properly. Please consider granting it this permission", Toast.LENGTH_LONG).show();
                }
            }
        }

    }

    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (requestCode == CAMERA_REQUEST && resultCode == RESULT_OK) {
            photo = (Bitmap) data.getExtras().get("data");
            imageView.setImageBitmap(photo);
            if(photo==null) {
                Log.i("on activity method", "on activity executed");
            }else {
                Log.i("on activity method", "noo activity executed");
            }
            display(photo);
        }
    }
}
