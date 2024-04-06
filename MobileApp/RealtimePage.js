import React, { useEffect, useState } from 'react';
import { Modal, View, Text, TouchableOpacity, StyleSheet, ScrollView, Pressable } from 'react-native';



const RealtimePage = ({tdata}) => {
    const [modalVisible, setModalVisible] = useState(false);
    const [modalData, setModalData] = useState([]);


    const categories = [
        { name: Object.keys(tdata)[1], color: '#e7dd14', count: tdata.burning.count, durations: tdata.burning.arrDur },
        { name: Object.keys(tdata)[2], color: '#e7dd14', count: tdata.coughing.count, durations: tdata.coughing.arrDur },
        { name: Object.keys(tdata)[3], color: '#e7dd14', count: tdata.gasStove.count, durations: tdata.gasStove.arrDur },
        { name: Object.keys(tdata)[4], color: '#e7dd14', count: tdata.glass_breaking.count, durations: tdata.glass_breaking.arrDur },
        { name: Object.keys(tdata)[5], color: '#e7dd14', count: tdata.plasticCollapse.count, durations: tdata.plasticCollapse.arrDur },
        { name: Object.keys(tdata)[6], color: '#e7dd14', count: tdata.runningTapWater.count, durations: tdata.runningTapWater.arrDur },
        { name: Object.keys(tdata)[7], color: '#e7dd14', count: tdata.sneezing.count, durations: tdata.sneezing.arrDur },
        { name: Object.keys(tdata)[8], color: '#e7dd14', count: tdata.sniffingNose.count, durations: tdata.sniffingNose.arrDur }
    ];

    const ClassCategoryIcons = () => {
        return (
            <View style={styles.container}>
            {categories.map((category, index) => (
                <TouchableOpacity
                key={index}
                style={[styles.iconContainer, { backgroundColor: category.color }]}
                onPress={() => handleIconPress(category)}
                >
                <Text>{category.name}</Text>
                <Text>{category.count}</Text>
                </TouchableOpacity>
            ))}
            </View>
        );
    };
    const handleIconPress = (c) => {
        setModalData(Object.values(c.durations));
        setModalVisible(true);
    };
    const styles = StyleSheet.create({
        container: {
            display: 'flex',
            flexDirection: 'row',
            flexWrap: 'wrap',
            width: '100%',
            justifyContent: 'center',
            paddingHorizontal: 10,
            paddingVertical: 10,
            backgroundColor: 'white'
        },
        iconContainer: {
            width: '46%',
            aspectRatio : '1 / 1',
            borderRadius: 30,
            justifyContent: 'center',
            alignItems: 'center',
            paddingHorizontal: 10,
            paddingVertical: 10,
            margin: 5
        },
        icon: {
            fontSize: 24,
            color: 'white',
        },
        modalView: {
            margin: 20,
            backgroundColor: 'white',
            borderRadius: 20,
            padding: 35,
            alignItems: 'center',
            shadowColor: '#000',
            shadowOffset: {
                width: 0,
                height: 2,
            },
            shadowOpacity: 0.25,
            shadowRadius: 4,
            elevation: 5,
        },
        centeredView: {
            flex: 1,
            justifyContent: 'center',
            alignItems: 'center',
            marginTop: 22,
        }
    });
    return (
        <ScrollView>
            <ClassCategoryIcons></ClassCategoryIcons>
            <Modal visible={modalVisible} animationType="slide" transparent={true}>
                <View style={styles.centeredView}>
                    <View style={styles.modalView}>
                        <Text>Duration of each time:</Text>
                        {modalData.map((info, index) => <Text key={index}>{info}</Text>)}
                        <Pressable onPress={() => setModalVisible(!modalVisible)}>
                            <Text>Got it</Text>
                        </Pressable>
                    </View>
                </View>
            </Modal>
        </ScrollView>
    );
};

export default RealtimePage;