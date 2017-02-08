package main;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.MultipleInputs;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashSet;
import java.util.Set;
import java.util.StringTokenizer;

public class CommonWords {

    private static String stopWordPath;

    // tokenize file 1

    /**
     * Generate the Map<Text, Int> for WordCount for File1
     *
     */
    public static class TokenizerMapperOne
            extends Mapper<Object, Text, Text, IntWritable> {

        public Set<String> stopwords = new HashSet<>();

        /*
         * read stopword list
         */
        @Override
        protected void setup(Context context) {
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

        private final static IntWritable one = new IntWritable(1);
        private Text word = new Text();

        public void map(Object key, Text value, Context context
        ) throws IOException, InterruptedException {
            StringTokenizer itr = new StringTokenizer(value.toString());
            while (itr.hasMoreTokens()) {
                word.set(itr.nextToken());
                if (stopwords.contains(word.toString()))
                    continue;
                context.write(word, one);
            }
        }
    }

    public static class IntSumReducerOne
            extends Reducer<Text, IntWritable, Text, Text> {
        private IntWritable result = new IntWritable();

        public void reduce(Text key, Iterable<IntWritable> values,
                           Context context
        ) throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            result.set(sum);
            context.write(key, new Text(result.toString()));
        }
    }


    /**
     * Generate the Map<Text, Int> for WordCount for File2
     *
     */
    public static class TokenizerMapperTwo
            extends Mapper<Object, Text, Text, IntWritable> {

        public Set<String> stopwords = new HashSet<String>();

        /*
         * read stopword list
         */
        @Override
        protected void setup(Context context) {
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


        private final static IntWritable one = new IntWritable(1);
        private Text word = new Text();

        public void map(Object key, Text value, Context context
        ) throws IOException, InterruptedException {
            StringTokenizer itr = new StringTokenizer(value.toString());
            while (itr.hasMoreTokens()) {
                word.set(itr.nextToken());
                if (stopwords.contains(word.toString()))
                    continue;
                context.write(word, one);
            }
        }
    }

    public static class IntSumReducerTwo
            extends Reducer<Text, IntWritable, Text, Text> {
        private IntWritable result = new IntWritable();

        public void reduce(Text key, Iterable<IntWritable> values,
                           Context context
        ) throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            result.set(sum);
            context.write(key, new Text(result.toString()));
        }
    }


    /**
     * Generate the Map<Text, Frequency_Identifier>
     *
     */
    /**
     * TODO: change Key to Text, face the issue that the key is always the LineNumber, do not know how to overcome
     */
    public static class CombineMapper1
            extends Mapper<LongWritable, Text, Text, Text> {

        private Text word = new Text();

        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            //read one line, parse into (word, frequency) pair
            //output (word, frequency_s1)

            /**
             * The value is key\tvalue
             */
            String curValue = value.toString();
            String[] pair = curValue.split("\t");
            if (pair.length == 2) {
                String freqS1 = pair[1] + "_s1"; // add the mark for s1/s2
                context.write(new Text(pair[0]), new Text(freqS1));
            }
        }
    }

    public static class CombineMapper2
            extends Mapper<LongWritable, Text, Text, Text> {

        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            //read one line, parse into (word, frequency) pair
            //output (word, frequency_s2)
            String curValue = value.toString();
            String[] pair = curValue.split("\t");
            if (pair.length == 2) {
                String freqS2 = pair[1] + "_s2";
                context.write(new Text(pair[0]), new Text(freqS2));
            }
        }
    }

    public static class CombineReducer
            extends Reducer<Text, Text, Text, IntWritable> {

        public void reduce(Text key, Iterable<Text> value, Context context) throws IOException, InterruptedException {
            //parse each value (e.g., n1_s1), get frequency (n1) and stage identifier (s1)
            //if the key has two values, output (key, samller_frequency)
            //if the key has only one value, output nothing

            int count1 = 0;
            int count2 = 0;


            for (Text text : value) {
                String valueString = text.toString();
                String[] freqStage = valueString.split("_");

                if (freqStage.length != 2) {
                    continue;
                }

                if ("s1".equals(freqStage[1])) {
                    count1 = Integer.valueOf(freqStage[0]);

                } else if ("s2".equals(freqStage[1])) {
                    count2 = Integer.valueOf(freqStage[0]);
                }
            }
            //count the word frequence, if it is from file1 add one to count1, if it is from file2 add one to count2
            if (count1 != 0 && count2 != 0) {
                //write (key, smaller_frequence) out
                if (count1 < count2) {
                    context.write(key, new IntWritable(count1));
                } else {
                    context.write(key, new IntWritable(count2));
                }
            }
        }
    }


    /**
     * The sort step, just need to make the Int value as MapKey
     */
    /**
     * TODO: change Key to Text
     */
    public static class SortMapper
            extends Mapper<LongWritable, Text, IntWritable, Text> {

        IntWritable res = new IntWritable();

        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            //write the (value, key) pair
            String curValue = value.toString();
            String[] pair = curValue.split("\t");
            if (pair.length == 2) {
                context.write(new IntWritable(Integer.valueOf(pair[1])), new Text(pair[0]));
            }
            // context.write(new IntWritable(Integer.valueOf(valueString)), key);
        }
    }

    public static class SortReducer
            extends Reducer<IntWritable, Text, Text, IntWritable> {

        public void reduce(IntWritable key, Iterable<Text> value, Context context) throws IOException, InterruptedException {
            for (Text val : value) {
                //write (key, value) pair
                context.write(val, new IntWritable(Integer.valueOf(key.toString())));
            }
        }
    }

    public static class Comparator extends WritableComparator {
        public Comparator() {
            super(IntWritable.class, true);
            // TODO Auto-generated constructor stub
        }

        @Override
        public int compare(WritableComparable a, WritableComparable b) {
            return -super.compare(a, b);
        }
    }


    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();

        /**
         * Change to 7 input include the path of stopword, the output of stage 3 and 4 to make it more configurable
         */
        if (otherArgs.length != 7) {
            System.err.println("Usage: commonwords <input1> <output1> <input2> <output2> <stopword> <output3> <output4>");
            System.exit(2);
        }

        // extractStopwords();
        stopWordPath = otherArgs[4];

    /*
     * Do the word count for file one
     */
        Job job1 = new Job(conf, "wordcount 1");
        job1.setJarByClass(CommonWords.class);
        job1.setMapperClass(TokenizerMapperOne.class);
        job1.setReducerClass(IntSumReducerOne.class);
        job1.setMapOutputKeyClass(Text.class);
        job1.setMapOutputValueClass(IntWritable.class);
        job1.setOutputKeyClass(Text.class);
        job1.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(job1, new Path(otherArgs[0]));
        FileOutputFormat.setOutputPath(job1, new Path(otherArgs[1]));
        job1.waitForCompletion(true);

    /*
     * Do the word count for file two
     */
        //write you own job2
        Job job2 = new Job(conf, "wordcount 2");
        job2.setJarByClass(CommonWords.class);
        job2.setMapperClass(TokenizerMapperTwo.class);
        job2.setReducerClass(IntSumReducerTwo.class);
        job2.setMapOutputKeyClass(Text.class);
        job2.setMapOutputValueClass(IntWritable.class);
        job2.setOutputKeyClass(Text.class);
        job2.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(job2, new Path(otherArgs[2]));
        FileOutputFormat.setOutputPath(job2, new Path(otherArgs[3]));
        job2.waitForCompletion(true);
    /*
     * aggregate
     */

        Job job3 = new Job(conf, "aggregate");
        job3.setJarByClass(CommonWords.class);
        job3.setReducerClass(CombineReducer.class);
        job3.setMapOutputKeyClass(Text.class);
        job3.setMapOutputValueClass(Text.class);
        job3.setOutputKeyClass(Text.class);
        job3.setOutputValueClass(IntWritable.class);
        MultipleInputs.addInputPath(job3, new Path(otherArgs[1]), job1.getInputFormatClass(), CombineMapper1.class);
        MultipleInputs.addInputPath(job3, new Path(otherArgs[3]),
                job2.getInputFormatClass(), CombineMapper2.class);

        FileOutputFormat.setOutputPath(job3, new Path(otherArgs[5]));
        job3.waitForCompletion(true);


    /*
     * sorting
     */
        Job job4 = new Job(conf, "sorting");
        job4.setJarByClass(CommonWords.class);
        job4.setMapperClass(SortMapper.class);
        job4.setReducerClass(SortReducer.class);
        job4.setMapOutputKeyClass(IntWritable.class);
        job4.setMapOutputValueClass(Text.class);
        job4.setOutputKeyClass(Text.class);
        job4.setOutputValueClass(IntWritable.class);
        FileInputFormat.addInputPath(job4, new Path(otherArgs[5]));
        FileOutputFormat.setOutputPath(job4, new Path(otherArgs[6]));
        //write your won job 4 and add the following sentence before job4.waitForCompletion(true)
        job4.setSortComparatorClass(Comparator.class);
        job4.waitForCompletion(true);


    }
}
