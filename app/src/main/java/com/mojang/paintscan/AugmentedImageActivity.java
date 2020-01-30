
package com.mojang.paintscan;

import android.Manifest;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.os.Bundle;
import android.os.Environment;
import android.support.design.widget.FloatingActionButton;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.Gravity;
import android.view.MotionEvent;
import android.view.View;
import android.widget.ImageView;
import android.widget.Toast;

import com.google.ar.core.Anchor;
import com.google.ar.core.AugmentedImage;
import com.google.ar.core.Camera;
import com.google.ar.core.Frame;
import com.google.ar.core.HitResult;
import com.google.ar.core.Plane;
import com.google.ar.core.Pose;
import com.google.ar.core.exceptions.NotYetAvailableException;
import com.google.ar.sceneform.AnchorNode;
import com.google.ar.sceneform.ArSceneView;
import com.google.ar.sceneform.FrameTime;
import com.google.ar.sceneform.Node;
import com.google.ar.sceneform.math.Matrix;
import com.google.ar.sceneform.math.Vector3;
import com.google.ar.sceneform.rendering.Color;
import com.google.ar.sceneform.rendering.MaterialFactory;
import com.google.ar.sceneform.rendering.ModelRenderable;
import com.google.ar.sceneform.rendering.ShapeFactory;
import com.mojang.common.helpers.SnackbarHelper;
import com.google.ar.sceneform.ux.ArFragment;
import com.google.ar.sceneform.ux.TransformableNode;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;

import java.io.BufferedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * This application demonstrates using augmented images to place anchor nodes. app to include image
 * tracking functionality.
 *
 * <p>In this example, we assume all images are static or moving slowly with a large occupation of
 * the screen. If the target is actively moving, we recommend to check
 * ArAugmentedImage_getTrackingMethod() and render only when the tracking method equals to
 * AR_AUGMENTED_IMAGE_TRACKING_METHOD_FULL_TRACKING. See details in <a
 * href="https://developers.google.com/ar/develop/c/augmented-images/">Recognize and Augment
 * Images</a>.
 */
public class AugmentedImageActivity extends AppCompatActivity {

  static {
    System.loadLibrary("opencv_java3");
  }

  static final String LOG_TAG = "AugmentedImageActivity";

  private ArFragment mARFragment = null;
  private ImageView mFitToScanView = null;
  private Frame mFrame = null;
  private AugmentedImage mAugmentedImage = null;
  private ModelRenderable mSphereRenderable = null;

  private Node[] mReferenceNodes = null;

  // Augmented image and its associated center pose anchor, keyed by the augmented image in
  // the database.
  private final Map<AugmentedImage, AugmentedImageNode> augmentedImageMap = new HashMap<>();

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);

    if (!OpenCVLoader.initDebug()) {
      Log.e(LOG_TAG, "OpenCVLoader initDebug failed.");
    }

    setContentView(R.layout.activity_main);

    createPhotoOutputFolder();

    mARFragment = (ArFragment) getSupportFragmentManager().findFragmentById(R.id.ux_fragment);
    mFitToScanView = findViewById(R.id.image_view_fit_to_scan);

    mARFragment.getArSceneView().getScene().addOnUpdateListener(this::onUpdateFrame);

    ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE},1);

    MaterialFactory.makeOpaqueWithColor(this, new Color(android.graphics.Color.RED))
            .thenAccept(material -> {
              mSphereRenderable = ShapeFactory.makeSphere(0.02f, new Vector3(0.0f, 0.0f, 0.0f), material);
            });

    FloatingActionButton fab = findViewById(R.id.floatingActionButton);
    fab.setOnClickListener(new View.OnClickListener() {
      @Override
      public void onClick(View view) {
        captureFrameImage(mFrame);
      }
    });
    fab.setEnabled(false);
  }

  private void captureFrameImage(Frame mFrame) {

    if (mFrame == null) {
      Log.e(LOG_TAG, "CaptureFrameImage called with null mFrame");
      return;
    }

    ArSceneView sceneView = mARFragment.getArSceneView();

    Image cameraImage = getARCoreImage(mFrame);
    byte[] cameraJpeg = extractImageDataFromARCore(cameraImage);
    String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
    String cameraFileName = "aOriginal_" + timeStamp + ".jpeg";
    saveImage(cameraJpeg, cameraFileName);

    com.google.ar.sceneform.Camera camera = sceneView.getScene().getCamera();
    Vector3 upperLeft = camera.worldToScreenPoint(mReferenceNodes[1].getWorldPosition());
    Vector3 upperRight = camera.worldToScreenPoint(mReferenceNodes[2].getWorldPosition());
    Vector3 lowerRight = camera.worldToScreenPoint(mReferenceNodes[3].getWorldPosition());
    Vector3 lowerLeft = camera.worldToScreenPoint(mReferenceNodes[4].getWorldPosition());

    final int sceneWidth = sceneView.getWidth();
    final int sceneHeight = sceneView.getHeight();

    final int cameraImageWidth = cameraImage.getWidth();
    final int cameraImageHeight = cameraImage.getHeight();

    List<Point> srcPts = new ArrayList<Point>();
    srcPts.add(new Point((upperLeft.y / sceneHeight) * cameraImageWidth, (1.0 - (upperLeft.x / sceneWidth)) * cameraImageHeight));
    srcPts.add(new Point((upperRight.y / sceneHeight) * cameraImageWidth, (1.0 - (upperRight.x / sceneWidth)) * cameraImageHeight));
    srcPts.add(new Point((lowerRight.y / sceneHeight) * cameraImageWidth, (1.0 - (lowerRight.x / sceneWidth)) * cameraImageHeight));
    srcPts.add(new Point((lowerLeft.y / sceneHeight) * cameraImageWidth, (1.0 - (lowerLeft.x / sceneWidth)) * cameraImageHeight));

//    srcPts.add(new Point((upperLeft.x / sceneWidth) * cameraImageHeight, (upperLeft.y / sceneHeight) * cameraImageWidth));
//    srcPts.add(new Point((upperRight.x / sceneWidth) * cameraImageHeight, (upperRight.y / sceneHeight) * cameraImageWidth));
//    srcPts.add(new Point((lowerRight.x / sceneWidth) * cameraImageHeight, (lowerRight.y / sceneHeight) * cameraImageWidth));
//    srcPts.add(new Point((lowerLeft.x / sceneWidth) * cameraImageHeight, (lowerLeft.y / sceneHeight) * cameraImageWidth));
//    srcPts.add(new Point((upperLeft.y / sceneHeight) * cameraImageHeight, (upperLeft.x / sceneWidth) * cameraImageWidth));
//    srcPts.add(new Point((upperRight.y / sceneHeight) * cameraImageHeight, (upperRight.x / sceneWidth) * cameraImageWidth));
//    srcPts.add(new Point((lowerRight.y / sceneHeight) * cameraImageHeight, (lowerRight.x / sceneWidth) * cameraImageWidth));
//    srcPts.add(new Point( (lowerLeft.y / sceneHeight) * cameraImageHeight, (lowerLeft.x / sceneWidth) * cameraImageWidth));

    // top-left, top-right, bottom-right, bottom-left
    List<Point> dstPoints = new ArrayList<Point>();
    dstPoints.add(new Point(0, 0));
    dstPoints.add(new Point(256, 0));
    dstPoints.add(new Point(256, 256));
    dstPoints.add(new Point(0, 256));

    Mat srcMat = Converters.vector_Point2f_to_Mat(srcPts);
    Mat dstMat = Converters.vector_Point2f_to_Mat(dstPoints);
    Mat perspectiveTransform = Imgproc.getPerspectiveTransform(srcMat, dstMat);

    //getting the input matrix from the given bitmap
    Bitmap cameraBitmap = BitmapFactory.decodeByteArray(cameraJpeg, 0, cameraJpeg.length);
    Mat inputMat = new Mat(cameraBitmap.getWidth(), cameraBitmap.getHeight(), CvType.CV_32S );
    Utils.bitmapToMat(cameraBitmap, inputMat);

    // MARK
    // A
    Imgproc.circle(inputMat, srcPts.get(0), 10, new Scalar(255, 0, 0), 5);
    Imgproc.circle(inputMat, srcPts.get(1), 10, new Scalar(0, 255, 0), 5);
    Imgproc.circle(inputMat, srcPts.get(2), 10, new Scalar(0, 0, 255), 5);
    Imgproc.circle(inputMat, srcPts.get(3), 10, new Scalar(125, 125, 0), 5);
    // B
    Imgproc.circle(inputMat, new Point(0, 0), 20, new Scalar(255, 0, 0), 10);
    Imgproc.circle(inputMat, new Point(cameraImageWidth, 0), 20, new Scalar(0, 255, 0), 10);
    Imgproc.circle(inputMat, new Point(cameraImageWidth, cameraImageHeight), 20, new Scalar(0, 0, 255), 10);
    Imgproc.circle(inputMat, new Point(0, cameraImageHeight), 20, new Scalar(125, 125, 0), 10);
    Utils.matToBitmap(inputMat, cameraBitmap);
    String markedFileName = "bMarked" + timeStamp + ".jpeg";
    saveImage(cameraBitmap, markedFileName);

    //getting the output matrix with the previously determined sizes
    Mat outputMat = new Mat(256, 256, CvType.CV_32S);

    //applying the transformation
    Imgproc.warpPerspective(inputMat, outputMat, perspectiveTransform, new Size(256, 256));

    //creating the output bitmap
    Bitmap outputBitmap = Bitmap.createBitmap(256, 256, Bitmap.Config.ARGB_8888);
    Utils.matToBitmap(outputMat, outputBitmap);

    String finalFileName = "final_" + timeStamp + ".jpeg";
    saveImage(outputBitmap, finalFileName);
  }

  private Image getARCoreImage(Frame frame) {
    Image cameraImage = null;
    try {
      cameraImage = frame.acquireCameraImage();
    } catch (NotYetAvailableException e) {
      e.printStackTrace();
      Log.e(LOG_TAG, "Failed to acquire camera image: " + e.getMessage());
      return null;
    }
    return cameraImage;
  }

//  private Bitmap getFrameBitmap(Frame frame) {
//    Image cameraImage = null;
//    try {
//      cameraImage = frame.acquireCameraImage();
//    } catch (NotYetAvailableException e) {
//      e.printStackTrace();
//      Log.e(LOG_TAG, "Failed to acquire camera image: " + e.getMessage());
//      return null;
//    }
//
//    // The camera image received is in YUV YCbCr Format. Get buffers for each of the planes and use them to create a new bytearray defined by the size of all three buffers combined
//    ByteBuffer cameraPlaneY = cameraImage.getPlanes()[0].getBuffer();
//    ByteBuffer cameraPlaneU = cameraImage.getPlanes()[1].getBuffer();
//    ByteBuffer cameraPlaneV = cameraImage.getPlanes()[2].getBuffer();
//
//    // Use the buffers to create a new byteArray that
//    byte[] compositeByteArray = new byte[cameraPlaneY.capacity() + cameraPlaneU.capacity() + cameraPlaneV.capacity()];
//
//    cameraPlaneY.get(compositeByteArray, 0, cameraPlaneY.capacity());
//    cameraPlaneU.get(compositeByteArray, cameraPlaneY.capacity(), cameraPlaneU.capacity());
//    cameraPlaneV.get(compositeByteArray, cameraPlaneY.capacity() + cameraPlaneU.capacity(), cameraPlaneV.capacity());
//
//    ByteArrayOutputStream baOutputStream = new ByteArrayOutputStream();
//    YuvImage yuvImage = new YuvImage(compositeByteArray, ImageFormat.NV21, cameraImage.getWidth(), cameraImage.getHeight(), null);
//    yuvImage.compressToJpeg(new Rect(0, 0, cameraImage.getWidth(), cameraImage.getHeight()), 100, baOutputStream);
//
//    byte[] byteForBitmap = baOutputStream.toByteArray();
//    return BitmapFactory.decodeByteArray(byteForBitmap, 0, byteForBitmap.length);
//  }

  @Override
  protected void onResume() {
    super.onResume();
    if (augmentedImageMap.isEmpty()) {
      mFitToScanView.setVisibility(View.VISIBLE);
    }
  }

  /**
   * Registered with the Sceneform Scene object, this method is called at the start of each frame.
   *
   * @param frameTime - time since last frame.
   */
  private void onUpdateFrame(FrameTime frameTime) {
    Frame frame = mARFragment.getArSceneView().getArFrame();

    // If there is no frame, just return.
    if (frame == null) {
      return;
    }

    mFrame = frame;

    FloatingActionButton fab = findViewById(R.id.floatingActionButton);

    Collection<AugmentedImage> updatedAugmentedImages = frame.getUpdatedTrackables(AugmentedImage.class);
    for (AugmentedImage augmentedImage : updatedAugmentedImages) {
      switch (augmentedImage.getTrackingState()) {
        case PAUSED:
          // When an image is in PAUSED state, but the camera is not PAUSED, it has been detected, but not yet tracked.
          Toast.makeText(this, "Image found, keep scanning.", Toast.LENGTH_SHORT);
          // String text = "Detected Image " + augmentedImage.getIndex();
          // SnackbarHelper.getInstance().showMessage(this, text);
          break;

        case TRACKING:
          // Have to switch to UI Thread to update View.
          mFitToScanView.setVisibility(View.GONE);

          // Create a new anchor for newly found images.
          if (!augmentedImageMap.containsKey(augmentedImage)) {
            AugmentedImageNode imageNode = new AugmentedImageNode(this);
            imageNode.setImage(augmentedImage);
            augmentedImageMap.put(augmentedImage, imageNode);
            mARFragment.getArSceneView().getScene().addChild(imageNode);

            mAugmentedImage = augmentedImage;
            if (mReferenceNodes != null) {
              for (Node rn : mReferenceNodes) {
                rn.setParent(null);
              }
            }
            mReferenceNodes = createReferenceNodes(mARFragment, mAugmentedImage);
          }

          fab.setEnabled(true);

          break;

        case STOPPED:
          augmentedImageMap.remove(augmentedImage);
          fab.setEnabled(false);
          break;
      }
    }
  }

  private Node[] createReferenceNodes(ArFragment fragment, AugmentedImage augmentedImage) {
    // Local cache
    float[] ullp = new float[]{-augmentedImage.getExtentX() / 2, 0f, -augmentedImage.getExtentZ() / 2};
    float[] urlp = new float[]{augmentedImage.getExtentX() / 2, 0f, -augmentedImage.getExtentZ() / 2};
    float[] lrlp = new float[]{augmentedImage.getExtentX() / 2, 0f, augmentedImage.getExtentZ() / 2};
    float[] lllp = new float[]{-augmentedImage.getExtentX() / 2, 0f, augmentedImage.getExtentZ() / 2};

//    Pose pose = mAugmentedImage.getCenterPose();
//    float[] ulwp = pose.transformPoint(ullp);
//    float[] urwp = pose.transformPoint(urlp);
//    float[] lrwp = pose.transformPoint(lrlp);
//    float[] llwp = pose.transformPoint(lllp);

    Node[] nodes = new Node[5];

    nodes[0] = new AnchorNode(mAugmentedImage.createAnchor(mAugmentedImage.getCenterPose()));
    fragment.getArSceneView().getScene().addChild(nodes[0]);

    nodes[1] = new Node();
    nodes[1].setLocalPosition(fromArray(ullp));
    nodes[1].setRenderable(mSphereRenderable);
    nodes[1].setParent(nodes[0]);

    nodes[2] = new Node();
    nodes[2].setWorldPosition(fromArray(urlp));
    nodes[2].setRenderable(mSphereRenderable);
    nodes[2].setParent(nodes[0]);

    nodes[3] = new Node();
    nodes[3].setWorldPosition(fromArray(lrlp));
    nodes[3].setRenderable(mSphereRenderable);
    nodes[3].setParent(nodes[0]);

    nodes[4] = new Node();
    nodes[4].setWorldPosition(fromArray(lllp));
    nodes[4].setRenderable(mSphereRenderable);
    nodes[4].setParent(nodes[0]);

    return nodes;
  }

  private static void createPhotoOutputFolder() {
    try {
      File file = generateSaveFile("foo");
      File folder = file.getParentFile();
      if (!folder.exists()) {
        if (!folder.mkdirs()) {
          throw new IOException();
        }
      }
    } catch (IOException e) {
      Log.e(LOG_TAG, "Failed to create saved_images directory: " + e.getMessage());
      e.printStackTrace();
    }
  }

  private static File generateSaveFile(String fileName) {
    return new File(Environment.getExternalStorageDirectory(), "/saved_images/" + fileName);
  }

  private static void saveImage(byte[] data, String fileName) {
    File file = generateSaveFile(fileName);
    if (file.exists()) {
      file.delete();
    }

    try {
      FileOutputStream out = new FileOutputStream(file);
      out.write(data);
      out.flush();
      out.close();
    } catch (IOException e) {
      Log.e(LOG_TAG, "Failed to save image: " + e.getMessage());
      e.printStackTrace();
    }
  }

  private static void saveImage(Bitmap bitmap, String fileName) {
    File file = generateSaveFile(fileName);
    if (file.exists()) {
      file.delete();
    }

    try {
      FileOutputStream out = new FileOutputStream(file);
      bitmap.compress(Bitmap.CompressFormat.JPEG, 100, out);
      out.flush();
      out.close();
    } catch (IOException e) {
      Log.e(LOG_TAG, "Failed to save image: " + e.getMessage());
      e.printStackTrace();
    }
  }

  private static Vector3 fromArray(float[] array) {
    float x = array.length > 0 ? array[0] : 0f;
    float y = array.length > 1 ? array[1] : 0f;
    float z = array.length > 2 ? array[2] : 0f;
    return new Vector3(x, y, z);
  }

  private static byte[] convertYUV420888toNV21(Image image) {
    byte[] nv21;
    ByteBuffer yBuffer = image.getPlanes()[0].getBuffer();
    ByteBuffer uBuffer = image.getPlanes()[1].getBuffer();
    ByteBuffer vBuffer = image.getPlanes()[2].getBuffer();

    int ySize = yBuffer.remaining();
    int uSize = uBuffer.remaining();
    int vSize = vBuffer.remaining();

    nv21 = new byte[ySize + uSize + vSize];

    //U and V are swapped
    yBuffer.get(nv21, 0, ySize);
    vBuffer.get(nv21, ySize, vSize);
    uBuffer.get(nv21, ySize + vSize, uSize);

    return nv21;
  }

  private static byte[] convertNV21toJPEG(byte[] nv21, int width, int height) {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    YuvImage yuv = new YuvImage(nv21, ImageFormat.NV21, width, height, null);
    yuv.compressToJpeg(new Rect(0, 0, width, height), 100, out);
    return out.toByteArray();
  }

  private static byte[] extractImageDataFromARCore(Image image) {
    byte[] nv21 = convertYUV420888toNV21(image);
    byte[] data = convertNV21toJPEG(nv21, image.getWidth(), image.getHeight());
    return data;

  }

}
