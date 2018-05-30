var express = require('express');
var router = express.Router();
var mysql = require('mysql');
var connection = mysql.createConnection({
    host: '',
    user: '',
    password: '',
    database: ''
});

connection.connect(function(error) {
    if (error) {
        console.log('Error');
    } else {
        console.log('Connected!');
    }
});

router.get('/', function(req, res, next) {
    res.send('Root REST API endpoint')
});

router.get('/list', function(req, res, next) {
    connection.query('SELECT * from biolitmap_data', function(err, rows, fields) {
        if (!err) {
            res.json(rows);
        } else {
            console.log('Error while performing query');
        }
    });
});

router.get('/filter/nameAffiliation/:name', function(req, res, next) {
    var name = req.params.name;
    name = "'" + name + "'";
    connection.query('SELECT * from biolitmap_data WHERE nameAffiliation = ' + name, function(err, rows, fields) {
        if (!err) {
            res.json(rows);
        } else {
            console.log('Error while performing query');
        }
    });
});

router.get('/filter/DOI/:doi', function(req, res, next) {
    var doi = req.params.doi;
    doi = "'" + doi + "'";
    connection.query('SELECT * from biolitmap_data WHERE DOI = ' + doi, function(err, rows, fields) {
        if (!err) {
            res.json(rows);
        } else {
            console.log('Error while performing query');
        }
    });
});

router.get('/filter/year/:year', function(req, res, next) {
    var year = req.params.year;
    if (year[0] == '+') {
        year = year.split("+")[1];
        connection.query('SELECT * from biolitmap_data WHERE year >= ' + year, function(err, rows, fields) {
            if (!err) {
                res.json(rows);
            } else {
                console.log('Error while performing query');
            }
        });
    } else if (year[0] == '-') {
        year = year.split("-")[1];
        connection.query('SELECT * from biolitmap_data WHERE year <= ' + year, function(err, rows, fields) {
            if (!err) {
                res.json(rows);
            } else {
                console.log('Error while performing query');
            }
        });
    } else {
        connection.query('SELECT * from biolitmap_data WHERE year = ' + year, function(err, rows, fields) {
            if (!err) {
                res.json(rows);
            } else {
                console.log('Error while performing query');
            }
        });
    }
});

router.get('/filter/source/:source', function(req, res, next) {
    var source = req.params.source;
    source = "'" + source + "'";
    connection.query('SELECT * from biolitmap_data WHERE source = ' + source, function(err, rows, fields) {
        if (!err) {
            res.json(rows);
        } else {
            console.log('Error while performing query');
        }
    });
});

router.get('/filter/edamCategory/:edamCategory', function(req, res, next) {
    var edamCategory = req.params.edamCategory;
    edamCategory = "'" + edamCategory + "'";
    connection.query('SELECT * from biolitmap_data WHERE edamCategory = ' + edamCategory, function(err, rows, fields) {
        if (!err) {
            res.json(rows);
        } else {
            console.log('Error while performing query');
        }
    });
});

module.exports = router;
