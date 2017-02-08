package main;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.input.MultipleInputs;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.lang.reflect.Method;
import java.util.*;

public class SearchDocuments {

    private static String stopWordPath;

    private static String keywordName = "keyword.txt";

    private static String dataFilesDirectory;

    private static HashSet<String> query = new HashSet<>();

    private static int totalFileNumber = 0;

    // Stage 1: Compute frequency of every word in a document
    // Mapper 1: (tokenize file)
    public static class TokenizerMapper extends
            Mapper<Object, Text, Text, IntWritable> {

        Set<String> stopwords = new HashSet<String>();

        @Override
        protected void setup(Context context) {
            //read in stopwords file
            //load stop words
            Configuration conf = context.getConfiguration();
            try {
                Path path = new Path(stopWordPath);
                FileSystem fs = FileSystem.get(new Configuration());
                BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(path)));
                String word = null;
                while ((word = br.readLine()) != null) {
                    stopwords.add(word);
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        private Text word_filename = new Text();
        private final static IntWritable one = new IntWritable(1);

        public void map(Object key, Text value, Context context)
                throws IOException, InterruptedException {

            // Get file name for a key/value pair in the Map function
            FileSplit fileSplit = getFileSplit(context);
            String fileName = fileSplit.getPath().getName().replace(".txt", "");
            // read one line. tokenize into (word@filename, 1) pairs
            StringTokenizer itr = new StringTokenizer(value.toString());
            while (itr.hasMoreTokens()) {
                Text word = new Text();
                word.set(itr.nextToken());
                if (stopwords.contains(word.toString()))
                    continue;
                word_filename = new Text(word.toString() + "@" + fileName);
                context.write(word_filename, one);
            }
        }
    }

    // Reducer 1: (calculate frequency of every word in every file)
    public static class IntSumReducer extends
            Reducer<Text, IntWritable, Text, IntWritable> {

        private IntWritable result = new IntWritable();

        public void reduce(Text key, Iterable<IntWritable> values,
                           Context context) throws IOException, InterruptedException {
            // sum up all the values, output (word@filename, freq) pair
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            result.set(sum);
            context.write(key, result);
        }
    }

    // Stage 2: Compute tf-idf of every word w.r.t. a document
    // Mapper 2:parse the output of stage1
    public static class Mapper2 extends Mapper<LongWritable, Text, Text, Text> {

        public void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {
            // parse the key/value pair into word, filename, frequency
            String curValue = value.toString();
            String[] pair = curValue.split("\t");
            if (pair.length == 2) {
                //I@d1
                String curKey = pair[0];
                //split I and d1
                String[] keyPair = curKey.split("@");
                if (keyPair.length == 2) {
                    //combine d1 and 1 to d1=1
                    String realValue = keyPair[1] + "=" + pair[1];
                    // output a pair (word, filename=frequency)
                    context.write(new Text(keyPair[0]), new Text(realValue));
                }
            }
        }
    }

    // Reducer 2: (calculate tf-idf of every word in every document)
    public static class Reducer2 extends Reducer<Text, Text, Text, Text> {
        public void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {

            //put <I@d1, 1> into Map
            Map<String, Integer> valueMap = new HashMap<>();
            int numberContains = 0;
            for (Text text: values) {
                // Note: key is a word, values are in the form of
                // (filename=frequency)
                String[] valuePair = text.toString().split("=");
                // sum up the number of files containing a particular word
                String realKey = key.toString() + "@" + valuePair[0];
                numberContains++;
                valueMap.put(realKey, Integer.valueOf(valuePair[1]));
            }
            // for every filename=frequency in the value, compute tf-idf of this
            // word in filename and output (word@filename, tfidf)
            for(Map.Entry<String, Integer> entry: valueMap.entrySet()) {
                double divided = ((double)totalFileNumber)/numberContains;
                double tdIdf = (1 + Math.log(entry.getValue())) * Math.log((divided));
                context.write(new Text(entry.getKey()), new Text(String.valueOf(tdIdf)));
            }
        }
    }

    // Stage 3: Compute normalized tf-idf
    // Mapper 3: (parse the output of stage 2)
    public static class Mapper3 extends Mapper<LongWritable, Text, Text, Text> {
        public void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {
            // parse the key/value pair into word, filename, tfidf
            String curValue = value.toString();
            String[] pair = curValue.split("\t");
            if (pair.length == 2) {
                //I@d1
                String curKey = pair[0];
                //split I and d1
                String[] keyPair = curKey.split("@");
                if (keyPair.length == 2) {
                    //combine d1 and 1 to d1=1
                    String realValue = keyPair[0] + "=" + pair[1];
                    // output a pair(filename, word=tfidf)
                    context.write(new Text(keyPair[1]), new Text(realValue));
                }
            }
        }
    }

    // Reducer 3: (compute normalized tf-idf of every word in very document)
    public static class Reducer3 extends Reducer<Text, Text, Text, Text> {
        public void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {
            // Note: key is a filename, values are in the form of (word=tfidf)
            //put <I@d1, tfidf> into Map
            Map<String, Double> valueMap = new HashMap<>();
            double totalNormValue = 0;
            for (Text text: values) {
                // Note: key is a word, values are in the form of
                // (filename=frequency)
                String[] valuePair = text.toString().split("=");
                // sum up the number of files containing a particular word
                String realKey =  valuePair[0] + "@" + key.toString();
                double curValue = Double.valueOf(valuePair[1]);
                totalNormValue = totalNormValue + curValue*curValue;
                valueMap.put(realKey, Double.valueOf(valuePair[1]));
            }

            double totalNormValueRoot = Math.sqrt(totalNormValue);
            // for every filename=frequency in the value, compute tf-idf of this
            // word in filename and output (word@filename, tfidf)
            for(Map.Entry<String, Double> entry: valueMap.entrySet()) {
                double normTfIdf = entry.getValue()/totalNormValueRoot;
                context.write(new Text(entry.getKey()), new Text(String.valueOf(normTfIdf)));
            }

        }
    }

    // Stage 4: Compute the relevance of every document w.r.t. a query
    // Mapper 4: (parse the output of stages)
    public static class Mapper4 extends Mapper<LongWritable, Text, Text, Text> {

        public void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {
            // parse the key/value pair into word, filename, norm-tfidf
            String curValue = value.toString();
            String[] pair = curValue.split("\t");
            if (pair.length == 2) {
                //I@d1
                String curKey = pair[0];
                //split I and d1
                String[] keyPair = curKey.split("@");
                if (keyPair.length == 2) {
                    // if the word is contained in the query file, output (filename,
                    // word=norm-tfidf)
                    if (query.contains(keyPair[0])) {
                        String realValue = keyPair[0] + "=" + pair[1];
                        // output a pair (word, filename=frequency)
                        context.write(new Text(keyPair[1]), new Text(realValue));
                    }
                }
            }
        }
    }

    // Reducer 4: (calculate relevance of every document w.r.t. the query)
    public static class Reducer4 extends Reducer<Text, Text, Text, Text> {
        public void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {
            double totalValue = 0;
            for (Text text: values) {
                // Note: key is a filename, values are in the form of
                // (word=norm-tfidf)
                String[] valuePair = text.toString().split("=");
                // sum up the number of files containing a particular word;
                double curValue = Double.valueOf(valuePair[1]);
                totalValue = totalValue + curValue;
            }

            context.write(key, new Text(String.valueOf(totalValue)));
        }
    }

    // Stage 5: Do the sorting here
    // Get local Top K
    public static class SortMapper extends
            Mapper<LongWritable, Text, DoubleWritable, Text> {

        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            //write the (value, key) pair
            String curValue = value.toString();
            String[] pair = curValue.split("\t");
            if (pair.length == 2) {
                context.write(new DoubleWritable(Double.valueOf(pair[1])), new Text(pair[0]));
            }
            // context.write(new IntWritable(Integer.valueOf(valueString)), key);
        }

    }

    public static class SortReducer extends
            Reducer<DoubleWritable, Text, Text, DoubleWritable> {

        public void reduce(DoubleWritable key, Iterable<Text> value, Context context) throws IOException, InterruptedException {
            for (Text val : value) {
                //write (key, value) pair
                context.write(val, new DoubleWritable(Double.valueOf(key.toString())));
            }
        }
    }

    public static class Comparator extends WritableComparator {
        public Comparator() {
            super(DoubleWritable.class, true);
            // TODO Auto-generated constructor stub
        }

        @Override
        public int compare(WritableComparable a, WritableComparable b) {
            return -super.compare(a, b);
        }
    }

    public static void main(String[] args) throws IOException,
            InterruptedException, ClassNotFoundException {
        Configuration conf = new Configuration();
        String[] otherArgs = new GenericOptionsParser(conf, args)
                .getRemainingArgs();
        if (otherArgs.length != 7) {
            System.err
                    .println("Usage: searchdocuments <documents directory> <output1> "
                            + "<output2> <output3> <output4> <output5> <stopword>");
            System.exit(2);
        }

        // extractStopwords();
        stopWordPath = otherArgs[6];

        // Stage 1: Compute frequency of every word in a document
        Job job1 = new Job(conf, "word-filename count");
        //set up Map Reduce function and Input Output Path
        job1.setJarByClass(SearchDocuments.class);
        job1.setMapperClass(TokenizerMapper.class);
        job1.setReducerClass(IntSumReducer.class);
        job1.setMapOutputKeyClass(Text.class);
        job1.setMapOutputValueClass(IntWritable.class);
        job1.setOutputKeyClass(Text.class);
        job1.setOutputValueClass(IntWritable.class);
        File directory = new File(otherArgs[0]);
        File[] files = directory.listFiles();
        for (File file: files) {
            if (!file.getName().equals(keywordName)) {
                totalFileNumber++;
                MultipleInputs.addInputPath(job1, new Path(file.getPath()), TextInputFormat.class, TokenizerMapper.class);
            }
        }

        FileOutputFormat.setOutputPath(job1, new Path(otherArgs[1]));
        job1.waitForCompletion(true);

        // Stage 2: Compute tf-idf of every word w.r.t. a document
        Job job2 = new Job(conf, "calculate tfidf");
        job2.setJarByClass(SearchDocuments.class);
        job2.setMapperClass(Mapper2.class);
        job2.setReducerClass(Reducer2.class);
        job2.setMapOutputKeyClass(Text.class);
        job2.setMapOutputValueClass(Text.class);
        job2.setOutputKeyClass(Text.class);
        job2.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(job2, new Path(otherArgs[1]));
        FileOutputFormat.setOutputPath(job2, new Path(otherArgs[2]));
        job2.waitForCompletion(true);


        // Stage 3: Compute normalized tf-idf
        Job job3 = new Job(conf, "nomalize tfidf");
        job3.setJarByClass(SearchDocuments.class);
        job3.setMapperClass(Mapper3.class);
        job3.setReducerClass(Reducer3.class);
        job3.setMapOutputKeyClass(Text.class);
        job3.setMapOutputValueClass(Text.class);
        job3.setOutputKeyClass(Text.class);
        job3.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(job3, new Path(otherArgs[2]));
        FileOutputFormat.setOutputPath(job3, new Path(otherArgs[3]));
        job3.waitForCompletion(true);

        // Stage 4: Compute the relevance of every document w.r.t. a query
        //set Query Directory
        dataFilesDirectory = otherArgs[0] + keywordName;
        try {
            Path path = new Path(dataFilesDirectory);
            FileSystem fs = FileSystem.get(new Configuration());
            BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(path)));
            String word = null;
            while ((word = br.readLine()) != null) {
                String[] wordList = word.split("\\s+");
                for (String keyWord: wordList) {
                    query.add(keyWord);
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        Job job4 = new Job(conf, "compute revelance");
        job4.setJarByClass(SearchDocuments.class);
        job4.setMapperClass(Mapper4.class);
        job4.setReducerClass(Reducer4.class);
        job4.setMapOutputKeyClass(Text.class);
        job4.setMapOutputValueClass(Text.class);
        job4.setOutputKeyClass(Text.class);
        job4.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(job4, new Path(otherArgs[3]));
        FileOutputFormat.setOutputPath(job4, new Path(otherArgs[4]));
        job4.waitForCompletion(true);


        // Stage 5: Get topk documents
        Job job5 = new Job(conf, "get top K documents");
        job5.setJarByClass(SearchDocuments.class);
        job5.setMapperClass(SortMapper.class);
        job5.setReducerClass(SortReducer.class);
        job5.setMapOutputKeyClass(DoubleWritable.class);
        job5.setMapOutputValueClass(Text.class);
        job5.setOutputKeyClass(Text.class);
        job5.setOutputValueClass(DoubleWritable.class);
        FileInputFormat.addInputPath(job5, new Path(otherArgs[4]));
        FileOutputFormat.setOutputPath(job5, new Path(otherArgs[5]));
        job5.setSortComparatorClass(Comparator.class);
        job5.waitForCompletion(true);
    }


    /**
     * Used for Step 1.
     * Deal with Error Message: ClassCastException:
     * org.apache.hadoop.mapreduce.lib.input.TaggedInputSplit cannot be cast to org.apache.hadoop.mapreduce.lib.input.FileSplit
     */
    private static FileSplit getFileSplit(Mapper.Context context) {
        InputSplit inputSplit = context.getInputSplit();
        Class<? extends InputSplit> splitClass = inputSplit.getClass();
        FileSplit fileSplit = null;
        if (splitClass.equals(FileSplit.class)) {
            fileSplit = (FileSplit) inputSplit;
        } else if (splitClass.getName().equals("org.apache.hadoop.mapreduce.lib.input.TaggedInputSplit")) {
            try {
                Method getInputSplitMethod = splitClass.getDeclaredMethod("getInputSplit");
                getInputSplitMethod.setAccessible(true);
                fileSplit = (FileSplit) getInputSplitMethod.invoke(inputSplit);
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        }
        return fileSplit;
    }

}