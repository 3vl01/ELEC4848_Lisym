import React, { useState, useEffect } from 'react';
import { View, Text, Button, ScrollView } from 'react-native';
import { LineChart } from 'react-native-chart-kit';
import {date} from './date';

// elaborate the number on health(respiratory system, limbs, mind)

const TrendsPage = ({data}) => {
    const [selectedDataset, setSelectedDataset] = useState('Past 7 days');
    const [selectedTrend, setSelectedTrend] = useState('Respiratory System');

    const handleDatasetChange = (dataset) => {
      setSelectedDataset(dataset);
    };

    const handleTrendChange = (trend) => {
        setSelectedTrend(trend);
    };

    const neededDates = Array.from({ length: 30 }, (_, i) => {
        const date = new Date();
        date.setDate(date.getDate() - i - 1);
        const year = date.getFullYear().toString().slice(-2);
        const month = (date.getMonth() + 1).toString().padStart(2, '0');
        const day = date.getDate().toString().padStart(2, '0');
        return `${year}${month}${day}`;
    });

    var arrayByDates = [];

    for (let day of neededDates){
        const da = new date(data, day);
        arrayByDates.push(da);
    }

    var dataset = {'Memory': {'Past 7 days': [0, 0, 0, 0, 0, 0, 0], 'Past month': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}, 'Respiratory System': {'Past 7 days': [0, 0, 0, 0, 0, 0, 0], 'Past month': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}, 'Limbs': {'Past 7 days': [0, 0, 0, 0, 0, 0, 0], 'Past month': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}};
    const classes = ['burning','coughing','gasStove','glass_breaking','plasticCollapse','runningTapWater','sneezing','sniffingNose'];

    var newClass = '';
    for (let clas of classes){
        if (clas == 'burning' || clas == 'gasStove' || clas == 'runningTapWater'){
            newClass = 'Memory';
        }
        else if (clas == 'coughing' || clas == 'sneezing' || clas == 'sniffingNose'){
            newClass = 'Respiratory System';
        }
        else{
            newClass = 'Limbs';
        }
        dataset[newClass] = {
            'Past 7 days': dataset[newClass]['Past 7 days'].map(function (num, idx) {
                return num + Array.from({ length: 7 }, (_, i) => {return arrayByDates[i][clas].count;})[idx];
              }),
            'Past month': dataset[newClass]['Past month'].map(function (num, idx) {
                return num + Array.from({ length: 30 }, (_, i) => {return arrayByDates[i][clas].count;})[idx];
            }),
        };
    }
    var message = '';
    var half = selectedDataset == 'Past 7 days'? 5:16;
    var upper = dataset[selectedTrend][selectedDataset].slice(half, dataset[selectedTrend][selectedDataset].length);
    var upperSum = 0;
    for (let i = 0; i < upper.length; i++) { upperSum += upper[i]; }
    var lower = dataset[selectedTrend][selectedDataset].slice(0, half);
    var lowerSum = 0;
    for (let i = 0; i < lower.length; i++) { lowerSum += lower[i]; }
    if (lowerSum > upperSum){
        message = "There is an increase in " + selectedTrend +" issues."; 
    }

    const labels = {
        'Past 7 days': Array.from({ length: 7 }, (_, i) => `Day ${i + 1}`),
        'Past month': Array.from({ length: 30 }, (_, i) => `Day ${i + 1}`),
    };
  
    return (
        <ScrollView>
            <Button
                title="Respiratory System"
                onPress={() => handleTrendChange('Respiratory System')}
                disabled={selectedTrend === 'Respiratory System'}
                color={'#e7dd14'}
            />
            <Button
                title="Limbs"
                onPress={() => handleTrendChange('Limbs')}
                disabled={selectedTrend === 'Limbs'}
                color={'#e7dd14'}
            />
            <Button
                title="Memory"
                onPress={() => handleTrendChange('Memory')}
                disabled={selectedTrend === 'Memory'}
                color={'#e7dd14'}
            />
            <View style={{ flex: 1, alignItems: 'center', justifyContent: 'center' }}>
                <View style={{ flexDirection: 'row', marginBottom: 20 }}>
                <Button
                    title="Past 7 days"
                    onPress={() => handleDatasetChange('Past 7 days')}
                    disabled={selectedDataset === 'Past 7 days'}
                    color={'#e7dd14'}
                />
                <Button
                    title="Past month"
                    onPress={() => handleDatasetChange('Past month')}
                    disabled={selectedDataset === 'Past month'}
                    color={'#e7dd14'}
                />
                </View>
                <Text style={{ marginBottom: 10 }}>Selected Dataset: {selectedDataset}</Text>

                <LineChart
                    data={{
                        labels: labels[selectedDataset],
                        datasets: [
                            {
                                data: dataset[selectedTrend][selectedDataset].reverse(),
                                color: (opacity = 1) => `rgba(255, 0, 0, ${opacity})`, // Red
                            },
                        ],
                    }}
                    width={400}
                    height={300}
                    yAxisLabel=""
                    verticalLabelRotation={90}
                    chartConfig={{
                        backgroundColor: '#ffffff',
                        backgroundGradientFrom: '#ffffff',
                        backgroundGradientTo: '#ffffff',
                        decimalPlaces: 0,
                        color: (opacity = 1) => `rgba(0, 0, 0, ${opacity})`,
                        labelColor: (opacity = 1) => `rgba(0, 0, 0, ${opacity})`,
                    }}
                    bezier
                />
            </View>
            <Text>{message}</Text>
        </ScrollView>
    );
};

export default TrendsPage;