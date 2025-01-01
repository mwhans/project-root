   // webpack.config.js
   const path = require('path');

   module.exports = {
     mode: 'development',
     entry: './src/mainController.ts',
     module: {
       rules: [
         {
           test: /\.tsx?$/,
           use: 'ts-loader',
           exclude: /node_modules/,
         },
       ],
     },
     resolve: {
       extensions: ['.tsx', '.ts', '.js'],
     },
     output: {
       filename: 'bundle.js',
       path: path.resolve(__dirname, 'dist'), // Adjust the output directory as needed
     },
     stats: {
       errorDetails: true,
     },
   };