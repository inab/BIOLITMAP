var http = require('http');
var express = require('express');
var indexRouter = require('./routes/index');
var biolitmapRouter = require('./routes/biolitmap');
var app = express();
var createError = require('http-errors');

app.use('/', indexRouter);
app.use('/biolitmap', biolitmapRouter);

// Catch 404 errors and forward them to the error handler
app.use(function(req, res, next) {
  next(createError(404));
});

module.exports = app;

app.listen(3002);
