package com.emaraic.ObjectRecognition;

import com.esotericsoftware.tablelayout.swing.Table;
import java.awt.Image;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.LinkedList;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JTextField;
import javax.swing.SwingUtilities;
import javax.swing.filechooser.FileNameExtensionFilter;
import org.tensorflow.DataType;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import java.io.BufferedReader;
import java.io.*;

/**
 *
 * @author Taha Emara Website: http://www.emaraic.com Email : taha@emaraic.com
 * Created on: Apr 29, 2017 Kindly: Don't remove this header Download the
 * pre-trained inception model from here:
 * https://storage.googleapis.com/download.tensorflow.org/models/inception_dec_2015.zip
 */
public class Recognizer extends JFrame implements ActionListener {

    private Table table;
    private JButton predictMlc;
    private JButton predictAni;
    private JButton incep;
    private JButton img;
    private JLabel viewer;
    private JTextField result;
    private JTextField imgpth;
    private JTextField modelpth;
    private FileNameExtensionFilter imgfilter = new FileNameExtensionFilter(
            "JPG & JPEG Images", "jpg", "jpeg");
    private String modelpath;
    private String imagepath;
    private byte[] graphDef;
    private List<String> labels;

    private List<String> paths = new LinkedList<>();

    public Recognizer() {
        setTitle("Object Recognition");
        setSize(500, 500);
        table = new Table();

        predictAni = new JButton("Predict Animals");
        predictMlc = new JButton("Predict MLC");
        predictAni.setEnabled(true);
        predictMlc.setEnabled(true);

        predictAni.addActionListener(this);
        predictMlc.addActionListener(this);

        viewer = new JLabel();
        getContentPane().add(table);

        table.addCell(predictAni).colspan(2);
        table.addCell(predictMlc).colspan(2);

        table.row();
        table.addCell(new JLabel("By: Tiger Cat")).center().padTop(30).colspan(2);
        table.row();
        table.addCell(new JLabel("Derived from Taha Emara")).center().colspan(2);

        setLocationRelativeTo(null);

        setResizable(false);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    } //Constructor
    
    
   // Lists all of the image files in the specified directory
    public void listFilesForFolder(File fold) {
        for (File fileEntry : fold.listFiles()) {
            if (fileEntry.isDirectory()) {
                listFilesForFolder(fileEntry);
            } else {
                System.out.println("File Name: " + fileEntry.getName());
                System.out.println("File Path: " + fileEntry.getPath());
                paths.add(fileEntry.getPath());
            }
        }
    }

    //MODIFACTIONS NEEDED FOR LOCAL EXECUTION
    //fold variable - path needs to be replaced with path to images for program to recognize (images should be in .jpg/jpeg format)
    //path variable - path needs to be replaced with path to output text file (file should be a blank .txt file)
    @Override
    public void actionPerformed(ActionEvent e) {
        paths = new LinkedList<>();
        
        predictAni.setEnabled(true);
        predictMlc.setEnabled(true);
        File fold;
        if (e.getSource() == predictMlc) {

            //path to images library
            //replace with path to images
            fold = new File("C:\\Users\\Nika_Kcin\\Documents\\UGA\\Object Recog\\object-recognition-tensorflow-master\\Resources\\images\\mlc");
        } else {
            //path to Images library
            //replace with path to images
            fold = new File("C:\\Users\\Nika_Kcin\\Documents\\UGA\\Object Recog\\object-recognition-tensorflow-master\\Resources\\images\\Animals");
        } //changing file directory
        
        
        try {
            //finds all of the image files
            listFilesForFolder(fold);
            
            //path to the output .txt file
            //replace with path to output text file
            String path = "C:\\Users\\Nika_Kcin\\Documents\\UGA\\Object Recog\\object-recognition-tensorflow-master\\src\\main\\java\\com\\emaraic\\ObjectRecognition\\Names.txt";

            BufferedWriter br = new BufferedWriter(new FileWriter(path));

            //Need to download inception_dec_2015
            //https://storage.googleapis.com/download.tensorflow.org/models/inception_dec_2015.zip
            modelpath = "C:\\Users\\Nika_Kcin\\Documents\\UGA\\Object Recog\\inception_dec_2015";
            System.out.println("Opening: " + modelpath);
            graphDef = readAllBytesOrExit(Paths.get(modelpath, "tensorflow_inception_graph.pb"));
            labels = readAllLinesOrExit(Paths.get(modelpath, "imagenet_comp_graph_label_strings.txt"));

            //path to spirit animal, Tiger Cat.
            imagepath = "C:\\Users\\Nika_Kcin\\Documents\\UGA\\Object Recog\\object-recognition-tensorflow-master\\Resources\\images\\cat.jpg";
            System.out.println("Image Path: " + imagepath);

            
            //reads the images and predicts image
            for (int i = 0; i < paths.size(); i++) {
                byte[] imageBytes = readAllBytesOrExit(Paths.get(imagepath));

                byte[] bytes = readAllBytesOrExit(Paths.get(paths.get(i)));

                try (Tensor image = Tensor.create(imageBytes)) {

                    Tensor img = Tensor.create(bytes);
                    float[] probs = executeInceptionGraph(graphDef, img);
                    int index = maxIndex(probs);

                    float[] labelProbabilities = executeInceptionGraph(graphDef, img);
                    int bestLabelIdx = maxIndex(labelProbabilities);
                    System.out.println(
                            String.format(
                                    "BEST MATCH: %s (%.2f%% likely)",
                                    labels.get(bestLabelIdx), labelProbabilities[bestLabelIdx] * 100f));

                    br.write(labels.get(index));
                    br.newLine();

                    System.out.println("Label: " + labels.get(index));

                } //try

            } //predict

            br.close();

        } catch (IOException err) {
            System.out.println("Stuff");
        } //catch

    } //ActionPerformed

    private static float[] executeInceptionGraph(byte[] graphDef, Tensor image) {
        try (Graph g = new Graph()) {
            g.importGraphDef(graphDef);
            try (Session s = new Session(g);
                    Tensor result = s.runner().feed("DecodeJpeg/contents", image).fetch("softmax").run().get(0)) {
                final long[] rshape = result.shape();
                if (result.numDimensions() != 2 || rshape[0] != 1) {
                    throw new RuntimeException(
                            String.format(
                                    "Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape %s",
                                    Arrays.toString(rshape)));
                }
                int nlabels = (int) rshape[1];
                return result.copyTo(new float[1][nlabels])[0];
            }
        }
    }

    private static int maxIndex(float[] probabilities) {
        int best = 0;
        for (int i = 1; i < probabilities.length; ++i) {
            if (probabilities[i] > probabilities[best]) {
                best = i;
            }
        }
        return best;
    }

    private static byte[] readAllBytesOrExit(Path path) {
        try {
            return Files.readAllBytes(path);
        } catch (IOException e) {
            System.err.println("Failed to read [" + path + "]: " + e.getMessage());
            System.exit(1);
        }
        return null;
    }

    private static List<String> readAllLinesOrExit(Path path) {
        try {
            return Files.readAllLines(path, Charset.forName("UTF-8"));
        } catch (IOException e) {
            System.err.println("Failed to read [" + path + "]: " + e.getMessage());
            System.exit(0);
        }
        return null;

    }

    // In the fullness of time, equivalents of the methods of this class should be auto-generated from
    // the OpDefs linked into libtensorflow_jni.so. That would match what is done in other languages
    // like Python, C++ and Go.
    static class GraphBuilder {

        GraphBuilder(Graph g) {
            this.g = g;
        }

        Output div(Output x, Output y) {
            return binaryOp("Div", x, y);
        }

        Output sub(Output x, Output y) {
            return binaryOp("Sub", x, y);
        }

        Output resizeBilinear(Output images, Output size) {
            return binaryOp("ResizeBilinear", images, size);
        }

        Output expandDims(Output input, Output dim) {
            return binaryOp("ExpandDims", input, dim);
        }

        Output cast(Output value, DataType dtype) {
            return g.opBuilder("Cast", "Cast").addInput(value).setAttr("DstT", dtype).build().output(0);
        }

        Output decodeJpeg(Output contents, long channels) {
            return g.opBuilder("DecodeJpeg", "DecodeJpeg")
                    .addInput(contents)
                    .setAttr("channels", channels)
                    .build()
                    .output(0);
        }

        Output constant(String name, Object value) {
            try (Tensor t = Tensor.create(value)) {
                return g.opBuilder("Const", name)
                        .setAttr("dtype", t.dataType())
                        .setAttr("value", t)
                        .build()
                        .output(0);
            }
        }

        private Output binaryOp(String type, Output in1, Output in2) {
            return g.opBuilder(type, type).addInput(in1).addInput(in2).build().output(0);
        }

        private Graph g;
    }

////////////
    public static void main(String[] args) {
        SwingUtilities.invokeLater(new Runnable() {
            public void run() {
                new Recognizer().setVisible(true);

            }
        });
    }

}
