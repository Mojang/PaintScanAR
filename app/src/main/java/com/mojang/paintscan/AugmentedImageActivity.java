
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
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;

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

    Bitmap cameraBitmap = getFrameBitmap(mFrame);

    com.google.ar.sceneform.Camera camera = mARFragment.getArSceneView().getScene().getCamera();

    Vector3 upperLeft = camera.worldToScreenPoint(mReferenceNodes[1].getWorldPosition());
    Vector3 upperRight = camera.worldToScreenPoint(mReferenceNodes[2].getWorldPosition());
    Vector3 lowerRight = camera.worldToScreenPoint(mReferenceNodes[3].getWorldPosition());
    Vector3 lowerLeft = camera.worldToScreenPoint(mReferenceNodes[4].getWorldPosition());

    // top-left, top-right, bottom-right, bottom-left
    List<Point> srcPts = new ArrayList<Point>();
    srcPts.add(new Point(upperLeft.x, upperLeft.y));
    srcPts.add(new Point(upperRight.x, upperRight.y));
    srcPts.add(new Point(lowerRight.x, lowerRight.y));
    srcPts.add(new Point(lowerLeft.x, lowerLeft.y));

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
    Mat inputMat = new Mat(cameraBitmap.getHeight(), cameraBitmap.getWidth(), CvType.CV_32F );
    Utils.bitmapToMat(cameraBitmap, inputMat);

    //getting the output matrix with the previously determined sizes
    Mat outputMat = new Mat(256, 256, CvType.CV_32F);

    //applying the transformation
    Imgproc.warpPerspective(inputMat, outputMat, perspectiveTransform, new Size(256, 256));
    //creating the output bitmap
    Bitmap outputBitmap = Bitmap.createBitmap(256, 256, Bitmap.Config.ARGB_8888);
    Utils.matToBitmap(outputMat, outputBitmap);

    saveImage(outputBitmap);
  }

  private Bitmap getFrameBitmap(Frame frame) {
    Image cameraImage = null;
    try {
      cameraImage = frame.acquireCameraImage();
    } catch (NotYetAvailableException e) {
      e.printStackTrace();
      Log.e(LOG_TAG, "Failed to acquire camera image: " + e.getMessage());
      return null;
    }

    // The camera image received is in YUV YCbCr Format. Get buffers for each of the planes and use them to create a new bytearray defined by the size of all three buffers combined
    ByteBuffer cameraPlaneY = cameraImage.getPlanes()[0].getBuffer();
    ByteBuffer cameraPlaneU = cameraImage.getPlanes()[1].getBuffer();
    ByteBuffer cameraPlaneV = cameraImage.getPlanes()[2].getBuffer();

    // Use the buffers to create a new byteArray that
    byte[] compositeByteArray = new byte[cameraPlaneY.capacity() + cameraPlaneU.capacity() + cameraPlaneV.capacity()];

    cameraPlaneY.get(compositeByteArray, 0, cameraPlaneY.capacity());
    cameraPlaneU.get(compositeByteArray, cameraPlaneY.capacity(), cameraPlaneU.capacity());
    cameraPlaneV.get(compositeByteArray, cameraPlaneY.capacity() + cameraPlaneU.capacity(), cameraPlaneV.capacity());

    ByteArrayOutputStream baOutputStream = new ByteArrayOutputStream();
    YuvImage yuvImage = new YuvImage(compositeByteArray, ImageFormat.NV21, cameraImage.getWidth(), cameraImage.getHeight(), null);
    yuvImage.compressToJpeg(new Rect(0, 0, cameraImage.getWidth(), cameraImage.getHeight()), 100, baOutputStream);

    byte[] byteForBitmap = baOutputStream.toByteArray();
    return BitmapFactory.decodeByteArray(byteForBitmap, 0, byteForBitmap.length);
  }

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

  private void createPhotoOutputFolder() {
    try {
      File file = generateSaveFile();
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

  private File generateSaveFile() {
    String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
    String fileName = "/saved_images/photo_" + timeStamp + ".jpeg";
    return new File(Environment.getExternalStorageDirectory(), fileName);
  }

  private void saveImage(Bitmap bitmap) {
    File file = generateSaveFile();
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

  private Vector3 fromArray(float[] array) {
    float x = array.length > 0 ? array[0] : 0f;
    float y = array.length > 1 ? array[1] : 0f;
    float z = array.length > 2 ? array[2] : 0f;
    return new Vector3(x, y, z);
  }
}
