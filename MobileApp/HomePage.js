import React, { useState } from 'react';
import { FlatList, Text, TouchableOpacity, View, StyleSheet, PanResponder, Pressable } from 'react-native';

var notiSwitchStove = false;
var notiSwitchWater = false;
var lastdivStove = 0;
var lastdivWater = 0;
var lastcountGlass = 0;
var lastcountPlastic = 0;
var lastcountBurning = 0;
var lastcountList = [lastcountGlass, lastcountPlastic, lastcountBurning];

const HomePage = ({tdata}) => {
    // noti page
    const [data, setData] = useState([]);

    const longDur = ({ event, threshold, notiSwitch, lastdiv }) => {
        var eventWord = "";
        if (event == "gasStove"){
            eventWord = "stove";
        }
        else if (event == "runningTapWater"){
            eventWord = "tap";
        }
        var time = new Date();
        const timeStr = ("0" + time.getHours()).slice(-2) + ("0" + time.getMinutes()).slice(-2) + ("0" + time.getSeconds()).slice(-2);
        const lastdur = tdata[event].arrDur.slice(-1);
        if (lastdur%threshold == 0 && lastdiv != Math.floor(lastdur/threshold)){
            notiSwitch = true;
        }
        if (notiSwitch == true){
            lastdiv = lastdur/threshold;
            var noti = {id: data.length, t: timeStr, warn: "The "+eventWord+" has been on for more than " +(threshold/60).toString()+ " minutes, is this normal?"};
            setData([noti, ...data]);
            notiSwitch = false;
        }
        happended({});
        return notiSwitch, lastdiv;
    }

    const happended = ({}) => {
        var event = "";
        var eventList = ["glass_breaking", "plasticCollapse", "burning"];
        var eventWord = "";
        var eventWordList = ["someone broke glass. Maybe related to human fall. Handle the shards cautiously.", "plastic has collapsed. Maybe related to human fall.", "something is burning."];
        var lastcount = 0;
        for (let i = 0; i < eventList.length; i++){
            event = eventList[i];
            eventWord = eventWordList[i];
            lastcount = lastcountList[i];
            var time = new Date();
            const timeStr = ("0" + time.getHours()).slice(-2) + ("0" + time.getMinutes()).slice(-2) + ("0" + time.getSeconds()).slice(-2);
            const count = tdata[event].count;
            if ((lastcount == 0 && count != 0) || count > lastcount){
                var noti = {id: data.length, t: timeStr, warn: "It seems that "+eventWord+" Please beware."};
                setData([noti, ...data]);
                lastcount = count;
            }
            lastcountList[i] = lastcount;
        }
    }
    
    notiSwitchStove, lastdivStove = longDur({ event: "gasStove", threshold: 300, notiSwitch: notiSwitchStove, lastdiv: lastdivStove });
    if (data.length != 0){
        notiSwitchWater, lastdivWater = longDur({ event: "runningTapWater", threshold: 60, notiSwitch: notiSwitchWater, lastdiv: lastdivWater });
    }

    const renderItem = ({ item }) => (
        <TouchableOpacity>
            <View style={styles.item}>
                <Text>{item.t}</Text>
                <Text>{item.warn}</Text>
            </View>
        </TouchableOpacity>
    );

    return (
        <FlatList
        data={data}
        renderItem={renderItem}
        keyExtractor={(item) => item.id}
        />
    );
};

const styles = StyleSheet.create({
    item: {
        padding: 20,
        borderBottomWidth: 1,
        borderBottomColor: '#ccc',
    },
});

export default HomePage;